#include "roller/loopclosure/cuda/leaf.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace roller::loopclosure::cuda {

    __device__ inline int spatialHashLeaf(int x, int y, int z, int table_size) {
        constexpr unsigned int p1 = 73856093;
        constexpr unsigned int p2 = 19349663;
        constexpr unsigned int p3 = 83492791;
        unsigned int h = (static_cast<unsigned int>(x) * p1) ^ 
                         (static_cast<unsigned int>(y) * p2) ^ 
                         (static_cast<unsigned int>(z) * p3);
        return static_cast<int>(h % static_cast<unsigned int>(table_size));
    }

    __global__ void buildSpatialHashKeysLeafKernel(
        const float3* d_surface,
        int surface_count,
        float cell_size,
        int table_size,
        int* d_cell_hashes,
        int* d_point_indices
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= surface_count) return;

        float3 pt = d_surface[idx];
        int cx = static_cast<int>(floorf(pt.x / cell_size));
        int cy = static_cast<int>(floorf(pt.y / cell_size));
        int cz = static_cast<int>(floorf(pt.z / cell_size));

        int hash = spatialHashLeaf(cx, cy, cz, table_size);
        d_cell_hashes[idx] = hash;
        d_point_indices[idx] = idx;
    }

    __global__ void buildCellBoundsLeafKernel(
        const int* d_cell_hashes,
        int surface_count,
        int table_size,
        int* d_cell_starts,
        int* d_cell_ends
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= surface_count) return;

        int hash = d_cell_hashes[idx];
        if (idx == 0) {
            d_cell_starts[hash] = 0;
        } else {
            int prev_hash = d_cell_hashes[idx - 1];
            if (hash != prev_hash) {
                d_cell_ends[prev_hash] = idx;
                d_cell_starts[hash] = idx;
            }
        }
        if (idx == surface_count - 1) {
            d_cell_ends[hash] = surface_count;
        }
    }

    __global__ void countLeafPointsSpatialHashKernel(
        const float3* d_surface,
        int surface_count,
        const int* d_cell_indices,
        const int* d_cell_starts,
        const int* d_cell_ends,
        int hash_table_size,
        const LeafNodeInfo* d_leafs,
        int leaf_count,
        float cell_size,
        FilledLeaf* d_filled_leafs_out
    ) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= leaf_count) return;

        LeafNodeInfo leaf = d_leafs[j];
        float3 center = leaf.get_center();
        float half_size = leaf.size * 0.5f;

        int min_cell_x = static_cast<int>(floorf((center.x - half_size) / cell_size));
        int max_cell_x = static_cast<int>(floorf((center.x + half_size) / cell_size));
        int min_cell_y = static_cast<int>(floorf((center.y - half_size) / cell_size));
        int max_cell_y = static_cast<int>(floorf((center.y + half_size) / cell_size));
        int min_cell_z = static_cast<int>(floorf((center.z - half_size) / cell_size));
        int max_cell_z = static_cast<int>(floorf((center.z + half_size) / cell_size));

        int count = 0;

        for (int cx = min_cell_x; cx <= max_cell_x; ++cx) {
            for (int cy = min_cell_y; cy <= max_cell_y; ++cy) {
                for (int cz = min_cell_z; cz <= max_cell_z; ++cz) {
                    int hash = spatialHashLeaf(cx, cy, cz, hash_table_size);
                    int start = d_cell_starts[hash];
                    int end = d_cell_ends[hash];

                    for (int k = start; k < end; ++k) {
                        int pt_idx = d_cell_indices[k];
                        if (pt_idx >= 0 && pt_idx < surface_count) {
                            float3 pt = d_surface[pt_idx];
                            if (fabsf(pt.x - center.x) <= half_size &&
                                fabsf(pt.y - center.y) <= half_size &&
                                fabsf(pt.z - center.z) <= half_size) {
                                count++;
                            }
                        }
                    }
                }
            }
        }

        FilledLeaf leaf_out;
        leaf_out.center = center;
        leaf_out.size = leaf.size;
        leaf_out.count = count;
        leaf_out.offset = 0;
        d_filled_leafs_out[j] = leaf_out;
    }

    __global__ void fillLeafPointsSpatialHashKernel(
        const float3* d_surface,
        int surface_count,
        const int* d_cell_indices,
        const int* d_cell_starts,
        const int* d_cell_ends,
        int hash_table_size,
        const LeafNodeInfo* d_leafs,
        int leaf_count,
        float cell_size,
        FilledLeaf* d_filled_leafs_out,
        int* d_point_indices_by_leaf_out
    ) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= leaf_count) return;

        LeafNodeInfo leaf = d_leafs[j];
        float3 center = leaf.get_center();
        float half_size = leaf.size * 0.5f;
        int offset = d_filled_leafs_out[j].offset;

        int min_cell_x = static_cast<int>(floorf((center.x - half_size) / cell_size));
        int max_cell_x = static_cast<int>(floorf((center.x + half_size) / cell_size));
        int min_cell_y = static_cast<int>(floorf((center.y - half_size) / cell_size));
        int max_cell_y = static_cast<int>(floorf((center.y + half_size) / cell_size));
        int min_cell_z = static_cast<int>(floorf((center.z - half_size) / cell_size));
        int max_cell_z = static_cast<int>(floorf((center.z + half_size) / cell_size));

        int written = 0;

        for (int cx = min_cell_x; cx <= max_cell_x; ++cx) {
            for (int cy = min_cell_y; cy <= max_cell_y; ++cy) {
                for (int cz = min_cell_z; cz <= max_cell_z; ++cz) {
                    int hash = spatialHashLeaf(cx, cy, cz, hash_table_size);
                    int start = d_cell_starts[hash];
                    int end = d_cell_ends[hash];

                    for (int k = start; k < end; ++k) {
                        int pt_idx = d_cell_indices[k];
                        if (pt_idx >= 0 && pt_idx < surface_count) {
                            float3 pt = d_surface[pt_idx];
                            if (fabsf(pt.x - center.x) <= half_size &&
                                fabsf(pt.y - center.y) <= half_size &&
                                fabsf(pt.z - center.z) <= half_size) {
                                d_point_indices_by_leaf_out[offset + written] = pt_idx;
                                written++;
                            }
                        }
                    }
                }
            }
        }
    }

    void attributePointsToLeafsGPU(
        const float3* d_surface,
        int surface_count,
        const LeafNodeInfo* d_leafs,
        int leaf_count,
        FilledLeaf* d_filled_leafs_out,
        int** d_point_indices_by_leaf_out,
        int& total_leaf_points_out,
        cudaStream_t stream
    ) {
        total_leaf_points_out = 0;
        if (surface_count <= 0 || leaf_count <= 0 || !d_surface || !d_leafs || !d_filled_leafs_out) return;

        float cell_size = 2.0f;
        int hash_table_size = surface_count * 2;
        if (hash_table_size < 1024) hash_table_size = 1024;

        thrust::device_vector<int> d_cell_hashes(surface_count);
        thrust::device_vector<int> d_point_indices(surface_count);
        thrust::device_vector<int> d_cell_starts(hash_table_size, -1);
        thrust::device_vector<int> d_cell_ends(hash_table_size, -1);

        int threads = 256;
        int surf_blocks = (surface_count + threads - 1) / threads;

        buildSpatialHashKeysLeafKernel<<<surf_blocks, threads, 0, stream>>>(
            d_surface, surface_count, cell_size, hash_table_size,
            thrust::raw_pointer_cast(d_cell_hashes.data()),
            thrust::raw_pointer_cast(d_point_indices.data())
        );

        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            d_cell_hashes.begin(), d_cell_hashes.end(),
            d_point_indices.begin()
        );

        buildCellBoundsLeafKernel<<<surf_blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(d_cell_hashes.data()),
            surface_count, hash_table_size,
            thrust::raw_pointer_cast(d_cell_starts.data()),
            thrust::raw_pointer_cast(d_cell_ends.data())
        );

        int leaf_blocks = (leaf_count + threads - 1) / threads;

        countLeafPointsSpatialHashKernel<<<leaf_blocks, threads, 0, stream>>>(
            d_surface, surface_count,
            thrust::raw_pointer_cast(d_point_indices.data()),
            thrust::raw_pointer_cast(d_cell_starts.data()),
            thrust::raw_pointer_cast(d_cell_ends.data()),
            hash_table_size, d_leafs, leaf_count, cell_size, d_filled_leafs_out
        );
        cudaStreamSynchronize(stream);

        thrust::device_vector<int> d_counts(leaf_count);
        thrust::device_vector<int> d_offsets(leaf_count);

        thrust::transform(
            thrust::cuda::par.on(stream),
            d_filled_leafs_out, d_filled_leafs_out + leaf_count,
            d_counts.begin(),
            [] __device__ (const FilledLeaf& l) { return l.count; }
        );

        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            d_counts.begin(), d_counts.end(),
            d_offsets.begin()
        );

        thrust::transform(
            thrust::cuda::par.on(stream),
            d_filled_leafs_out, d_filled_leafs_out + leaf_count,
            d_offsets.begin(),
            d_filled_leafs_out,
            [] __device__ (FilledLeaf l, int off) { l.offset = off; return l; }
        );

        int last_count = d_counts.back();
        int last_offset = d_offsets.back();
        total_leaf_points_out = last_offset + last_count;

        if (total_leaf_points_out > 0 && d_point_indices_by_leaf_out) {
            cudaMalloc(d_point_indices_by_leaf_out, total_leaf_points_out * sizeof(int));
            fillLeafPointsSpatialHashKernel<<<leaf_blocks, threads, 0, stream>>>(
                d_surface, surface_count,
                thrust::raw_pointer_cast(d_point_indices.data()),
                thrust::raw_pointer_cast(d_cell_starts.data()),
                thrust::raw_pointer_cast(d_cell_ends.data()),
                hash_table_size, d_leafs, leaf_count, cell_size,
                d_filled_leafs_out, *d_point_indices_by_leaf_out
            );
        }
    }

} // namespace roller::loopclosure::cuda
