#include "roller/loopclosure/cuda/radius_search.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <cmath>

namespace roller::loopclosure::cuda {

    __device__ inline uint32_t computeSpatialHash(int vx, int vy, int vz, int hash_table_size) {
        constexpr uint32_t p1 = 73856093u;
        constexpr uint32_t p2 = 19349663u;
        constexpr uint32_t p3 = 83492791u;
        uint32_t hash = ((static_cast<uint32_t>(vx) * p1) ^ 
                         (static_cast<uint32_t>(vy) * p2) ^ 
                         (static_cast<uint32_t>(vz) * p3));
        return hash % static_cast<uint32_t>(hash_table_size);
    }

    __global__ void computeCellHashKeysKernel(
        const float3* d_surface,
        int surface_count,
        float cell_size,
        int hash_table_size,
        uint32_t* d_keys_out,
        uint32_t* d_indices_out
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= surface_count) return;

        float3 pt = d_surface[idx];
        int vx = static_cast<int>(floorf(pt.x / cell_size));
        int vy = static_cast<int>(floorf(pt.y / cell_size));
        int vz = static_cast<int>(floorf(pt.z / cell_size));

        d_keys_out[idx] = computeSpatialHash(vx, vy, vz, hash_table_size);
        d_indices_out[idx] = static_cast<uint32_t>(idx);
    }

    __global__ void buildCellBoundsKernel(
        const uint32_t* d_sorted_keys,
        int surface_count,
        int* d_cell_starts,
        int* d_cell_ends
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= surface_count) return;

        uint32_t current_key = d_sorted_keys[idx];
        if (idx == 0) {
            d_cell_starts[current_key] = 0;
        } else {
            uint32_t prev_key = d_sorted_keys[idx - 1];
            if (current_key != prev_key) {
                d_cell_starts[current_key] = idx;
                d_cell_ends[prev_key] = idx;
            }
        }
        if (idx == surface_count - 1) {
            d_cell_ends[current_key] = surface_count;
        }
    }

    __global__ void countPatchPointsSpatialHashKernel(
        const float3* d_surface,
        int surface_count,
        const uint32_t* d_sorted_surface_indices,
        const int* d_cell_starts,
        const int* d_cell_ends,
        int hash_table_size,
        const float3* d_query,
        int query_count,
        float radius,
        Patch* d_patches_out
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= query_count) return;

        float3 q_pt = d_query[idx];
        int count = 0;

        float radius_sq = radius * radius;
        float cell_size = radius;

        int center_vx = static_cast<int>(floorf(q_pt.x / cell_size));
        int center_vy = static_cast<int>(floorf(q_pt.y / cell_size));
        int center_vz = static_cast<int>(floorf(q_pt.z / cell_size));

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uint32_t hash = computeSpatialHash(center_vx + dx, center_vy + dy, center_vz + dz, hash_table_size);
                    int start = d_cell_starts[hash];
                    if (start == -1) continue;
                    int end = d_cell_ends[hash];

                    for (int i = start; i < end; ++i) {
                        uint32_t surf_idx = d_sorted_surface_indices[i];
                        float3 s_pt = d_surface[surf_idx];

                        float dx_p = q_pt.x - s_pt.x;
                        float dy_p = q_pt.y - s_pt.y;
                        float dz_p = q_pt.z - s_pt.z;
                        float dist_sq = dx_p * dx_p + dy_p * dy_p + dz_p * dz_p;

                        if (dist_sq <= radius_sq) {
                            count++;
                        }
                    }
                }
            }
        }

        Patch patch;
        patch.query_pt = q_pt;
        patch.count = count;
        patch.offset = 0;
        d_patches_out[idx] = patch;
    }

    __global__ void fillPatchIndicesSpatialHashKernel(
        const float3* d_surface,
        int surface_count,
        const uint32_t* d_sorted_surface_indices,
        const int* d_cell_starts,
        const int* d_cell_ends,
        int hash_table_size,
        const float3* d_query,
        int query_count,
        float radius,
        const Patch* d_patches_in,
        int* d_patch_indices_out
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= query_count) return;

        Patch patch = d_patches_in[idx];
        if (patch.count <= 0) return;

        float3 q_pt = patch.query_pt;
        int write_pos = patch.offset;

        float radius_sq = radius * radius;
        float cell_size = radius;

        int center_vx = static_cast<int>(floorf(q_pt.x / cell_size));
        int center_vy = static_cast<int>(floorf(q_pt.y / cell_size));
        int center_vz = static_cast<int>(floorf(q_pt.z / cell_size));

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uint32_t hash = computeSpatialHash(center_vx + dx, center_vy + dy, center_vz + dz, hash_table_size);
                    int start = d_cell_starts[hash];
                    if (start == -1) continue;
                    int end = d_cell_ends[hash];

                    for (int i = start; i < end; ++i) {
                        uint32_t surf_idx = d_sorted_surface_indices[i];
                        float3 s_pt = d_surface[surf_idx];

                        float dx_p = q_pt.x - s_pt.x;
                        float dy_p = q_pt.y - s_pt.y;
                        float dz_p = q_pt.z - s_pt.z;
                        float dist_sq = dx_p * dx_p + dy_p * dy_p + dz_p * dz_p;

                        if (dist_sq <= radius_sq) {
                            d_patch_indices_out[write_pos++] = static_cast<int>(surf_idx);
                        }
                    }
                }
            }
        }
    }

    struct GetCount {
        __host__ __device__ int operator()(const Patch& p) const { return p.count; }
    };

    struct SetOffset {
        __host__ __device__ Patch operator()(Patch p, int offset) const {
            p.offset = offset;
            return p;
        }
    };

    void performRadiusSearchGPU(
        const float3* d_surface,
        int surface_count,
        const float3* d_query,
        int query_count,
        float radius,
        Patch* d_patches_out,
        int** d_patch_indices_out,
        int& total_patch_points_out,
        cudaStream_t stream
    ) {
        total_patch_points_out = 0;
        if (d_patch_indices_out) *d_patch_indices_out = nullptr;
        if (query_count <= 0 || surface_count <= 0 || !d_surface || !d_query || !d_patches_out || radius <= 0.0f) return;

        int hash_table_size = 1 << static_cast<int>(std::ceil(std::log2(std::max(surface_count * 2, 1024))));

        thrust::device_vector<uint32_t> d_keys(surface_count);
        thrust::device_vector<uint32_t> d_indices(surface_count);

        uint32_t* raw_keys_ptr = thrust::raw_pointer_cast(d_keys.data());
        uint32_t* raw_indices_ptr = thrust::raw_pointer_cast(d_indices.data());

        int threads = 256;
        int blocks = (surface_count + threads - 1) / threads;

        computeCellHashKeysKernel<<<blocks, threads, 0, stream>>>(
            d_surface, surface_count, radius, hash_table_size, raw_keys_ptr, raw_indices_ptr
        );

        thrust::sort_by_key(thrust::cuda::par.on(stream), d_keys.begin(), d_keys.end(), d_indices.begin());

        thrust::device_vector<int> d_cell_starts(hash_table_size, -1);
        thrust::device_vector<int> d_cell_ends(hash_table_size, -1);

        int* raw_starts_ptr = thrust::raw_pointer_cast(d_cell_starts.data());
        int* raw_ends_ptr = thrust::raw_pointer_cast(d_cell_ends.data());

        buildCellBoundsKernel<<<blocks, threads, 0, stream>>>(
            raw_keys_ptr, surface_count, raw_starts_ptr, raw_ends_ptr
        );

        int query_blocks = (query_count + threads - 1) / threads;
        countPatchPointsSpatialHashKernel<<<query_blocks, threads, 0, stream>>>(
            d_surface, surface_count, raw_indices_ptr, raw_starts_ptr, raw_ends_ptr,
            hash_table_size, d_query, query_count, radius, d_patches_out
        );

        thrust::device_ptr<Patch> dev_patches_ptr(d_patches_out);
        thrust::device_vector<int> d_counts(query_count);
        thrust::transform(thrust::cuda::par.on(stream), dev_patches_ptr, dev_patches_ptr + query_count, d_counts.begin(), GetCount());

        thrust::device_vector<int> d_offsets(query_count);
        thrust::exclusive_scan(thrust::cuda::par.on(stream), d_counts.begin(), d_counts.end(), d_offsets.begin());

        thrust::transform(thrust::cuda::par.on(stream), dev_patches_ptr, dev_patches_ptr + query_count, d_offsets.begin(), dev_patches_ptr, SetOffset());

        int last_offset = d_offsets[query_count - 1];
        int last_count = d_counts[query_count - 1];
        total_patch_points_out = last_offset + last_count;

        if (total_patch_points_out > 0 && d_patch_indices_out) {
            cudaMalloc(d_patch_indices_out, total_patch_points_out * sizeof(int));
            fillPatchIndicesSpatialHashKernel<<<query_blocks, threads, 0, stream>>>(
                d_surface, surface_count, raw_indices_ptr, raw_starts_ptr, raw_ends_ptr,
                hash_table_size, d_query, query_count, radius, d_patches_out, *d_patch_indices_out
            );
        }
    }

} // namespace roller::loopclosure::cuda
