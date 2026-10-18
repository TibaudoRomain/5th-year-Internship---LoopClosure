#include "roller/loopclosure/cuda/pointcloud.cuh"
#include "roller/loopclosure/cuda/kernels.cuh"

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

namespace roller::loopclosure::cuda {

    __global__ void convertTSDFToFloat3Kernel(
        const nanovdb::NanoGrid<float>* sdf_grid,
        float3* d_points_out,
        int* d_point_counter,
        float max_distance_to_tsdf,
        float voxel_size,
        nanovdb::Coord bbox_min,
        int dimX, int dimY, int dimZ
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= dimX || y >= dimY || z >= dimZ) return;

        nanovdb::Coord xyz(x + bbox_min.x(), y + bbox_min.y(), z + bbox_min.z());
        auto acc = sdf_grid->getAccessor();

        if (!acc.isActive(xyz)) return;

        // Marching Cubes 8 corner sign crossing check
        int cube_index = 0;
        int active_corners = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            nanovdb::Coord corner = xyz + nanovdb::Coord((i & 1), ((i >> 1) & 1), ((i >> 2) & 1));
            if (acc.isActive(corner)) {
                active_corners++;
                float f = acc.getValue(corner);
                if (f < 0.0f) {
                    cube_index |= (1 << i);
                }
            }
        }

        if (active_corners < 4 || cube_index == 0 || cube_index == 255) return;

        float3 pos = make_float3(xyz.x() * voxel_size, xyz.y() * voxel_size, xyz.z() * voxel_size);
        int idx = atomicAdd(d_point_counter, 1);
        d_points_out[idx] = pos;
    }

    int convertTSDFToFloat3GPU(
        const nanovdb::NanoGrid<float>* d_sdf_grid,
        const nanovdb::CoordBBox& bbox,
        float voxel_size,
        float max_distance_to_tsdf,
        float3* d_points_out,
        int* d_point_counter,
        cudaStream_t stream
    ) {
        if (!d_sdf_grid || !d_points_out || !d_point_counter) return 0;

        cudaMemsetAsync(d_point_counter, 0, sizeof(int), stream);

        nanovdb::Coord bbox_min = bbox.min();
        nanovdb::Coord bbox_max = bbox.max();

        int dimX = bbox_max.x() - bbox_min.x() + 1;
        int dimY = bbox_max.y() - bbox_min.y() + 1;
        int dimZ = bbox_max.z() - bbox_min.z() + 1;

        if (dimX <= 0 || dimY <= 0 || dimZ <= 0) return 0;

        dim3 threads(8, 8, 8);
        dim3 blocks(
            (dimX + threads.x - 1) / threads.x,
            (dimY + threads.y - 1) / threads.y,
            (dimZ + threads.z - 1) / threads.z
        );

        convertTSDFToFloat3Kernel<<<blocks, threads, 0, stream>>>(
            d_sdf_grid, d_points_out, d_point_counter,
            max_distance_to_tsdf, voxel_size,
            bbox_min, dimX, dimY, dimZ
        );

        int host_counter = 0;
        cudaMemcpyAsync(&host_counter, d_point_counter, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        return host_counter;
    }

    __global__ void extractTSDFSurfaceKernel(
        const nanovdb::NanoGrid<float>* sdf_grid,
        PointNormal_GPU* dest,
        int* normalCounter,
        float max_distance_to_tsdf,
        float voxelSize,
        nanovdb::Coord bbox_min,
        int dimX, int dimY, int dimZ
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= dimX || y >= dimY || z >= dimZ) return;

        nanovdb::Coord xyz(x + bbox_min.x(), y + bbox_min.y(), z + bbox_min.z());
        auto acc = sdf_grid->getAccessor();

        if (!acc.isActive(xyz)) return;
        float val = acc.getValue(xyz);

        // Marching Cubes 8 corner sign crossing check: extract ONLY true surface zero-crossings
        int cube_index = 0;
        int active_corners = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            nanovdb::Coord corner = xyz + nanovdb::Coord((i & 1), ((i >> 1) & 1), ((i >> 2) & 1));
            if (acc.isActive(corner)) {
                active_corners++;
                float f = acc.getValue(corner);
                if (f < 0.0f) {
                    cube_index |= (1 << i);
                }
            }
        }

        if (active_corners < 4 || cube_index == 0 || cube_index == 255) return;

        // Prune isolated noise voxels with sparse 26-neighborhood density
        int active_neighbors = 0;
        #pragma unroll
        for (int n_dz = -1; n_dz <= 1; ++n_dz) {
            #pragma unroll
            for (int n_dy = -1; n_dy <= 1; ++n_dy) {
                #pragma unroll
                for (int n_dx = -1; n_dx <= 1; ++n_dx) {
                    if (n_dx == 0 && n_dy == 0 && n_dz == 0) continue;
                    if (acc.isActive(xyz + nanovdb::Coord(n_dx, n_dy, n_dz))) {
                        active_neighbors++;
                    }
                }
            }
        }
        if (active_neighbors < 3) return;

        bool is_normal_valid = false;
        float3 n = computeNormal(acc, xyz, voxelSize, is_normal_valid);
        if (!is_normal_valid || isnan(n.x) || isnan(n.y) || isnan(n.z)) {
            float val_px = acc.isActive(xyz.offsetBy(1, 0, 0)) ? acc.getValue(xyz.offsetBy(1, 0, 0)) : val;
            float val_nx = acc.isActive(xyz.offsetBy(-1, 0, 0)) ? acc.getValue(xyz.offsetBy(-1, 0, 0)) : val;
            float val_py = acc.isActive(xyz.offsetBy(0, 1, 0)) ? acc.getValue(xyz.offsetBy(0, 1, 0)) : val;
            float val_ny = acc.isActive(xyz.offsetBy(0, -1, 0)) ? acc.getValue(xyz.offsetBy(0, -1, 0)) : val;
            float val_pz = acc.isActive(xyz.offsetBy(0, 0, 1)) ? acc.getValue(xyz.offsetBy(0, 0, 1)) : val;
            float val_nz = acc.isActive(xyz.offsetBy(0, 0, -1)) ? acc.getValue(xyz.offsetBy(0, 0, -1)) : val;

            float dx = (val_px - val_nx) * 0.5f;
            float dy = (val_py - val_ny) * 0.5f;
            float dz = (val_pz - val_nz) * 0.5f;
            float len_sq = dx * dx + dy * dy + dz * dz;
            if (!isnan(len_sq) && len_sq > 1e-6f) {
                float inv_len = rsqrtf(len_sq);
                n = make_float3(dx * inv_len, dy * inv_len, dz * inv_len);
                is_normal_valid = true;
            } else {
                n = make_float3(0.0f, 0.0f, 1.0f);
            }
        }

        if (isnan(n.x) || isnan(n.y) || isnan(n.z)) {
            n = make_float3(0.0f, 0.0f, 1.0f);
        }

        PointNormal_GPU result;
        result.normal = n;
        result.pos = make_float3(xyz.x() * voxelSize, xyz.y() * voxelSize, xyz.z() * voxelSize);
        result.curvature = 0.0f;
        result.normal_trust = is_normal_valid ? 1.0f : 0.5f;

        int table_index = atomicAdd(normalCounter, 1);
        dest[table_index] = result;
    }

    int downsamplePointNormalBufferGPU(
        const PointNormal_GPU* d_input, int input_count, 
        PointNormal_GPU* d_output, float downsample_voxel_size
    ) {
        if (input_count <= 0 || !d_input || !d_output) return 0;

        if (downsample_voxel_size <= 0.0f) {
            cudaMemcpyAsync(d_output, d_input, input_count * sizeof(PointNormal_GPU), cudaMemcpyDeviceToDevice);
            return input_count;
        }

        thrust::device_vector<uint64_t> d_keys(input_count);
        thrust::device_vector<VoxelAccumulator> d_acc(input_count);

        uint64_t* raw_keys_ptr = thrust::raw_pointer_cast(d_keys.data());
        VoxelAccumulator* raw_acc_ptr = thrust::raw_pointer_cast(d_acc.data());

        int threads = 512;
        int blocks = (input_count + threads - 1) / threads;
        extractMonotonicVoxelKeysKernel<<<blocks, threads>>>(d_input, input_count, downsample_voxel_size, raw_keys_ptr, raw_acc_ptr);
        cudaDeviceSynchronize();

        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_acc.begin());

        thrust::device_vector<uint64_t> d_unique_keys(input_count);
        thrust::device_vector<VoxelAccumulator> d_reduced_acc(input_count);

        auto new_end = thrust::reduce_by_key(
            d_keys.begin(), d_keys.end(),
            d_acc.begin(),
            d_unique_keys.begin(),
            d_reduced_acc.begin()
        );

        int unique_count = static_cast<int>(new_end.first - d_unique_keys.begin());

        blocks = (unique_count + threads - 1) / threads;
        VoxelAccumulator* raw_reduced_acc_ptr = thrust::raw_pointer_cast(d_reduced_acc.data());
        finalizeVoxelCentroidsKernel<<<blocks, threads>>>(raw_reduced_acc_ptr, unique_count, d_output);
        cudaDeviceSynchronize();

        return unique_count;
    }

    __global__ void extractMonotonicVoxelKeysKernel(
        const PointNormal_GPU* points, int num_points, float voxel_size,
        uint64_t* keys_out, VoxelAccumulator* acc_out
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) return;

        PointNormal_GPU pt = points[idx];
        float3 pos = pt.pos;
        float3 n = pt.normal;
        if (isnan(n.x) || isnan(n.y) || isnan(n.z)) {
            n = make_float3(0.0f, 0.0f, 1.0f);
        }

        int vx = static_cast<int>(floorf(pos.x / voxel_size));
        int vy = static_cast<int>(floorf(pos.y / voxel_size));
        int vz = static_cast<int>(floorf(pos.z / voxel_size));

        uint64_t ux = static_cast<uint64_t>(vx + 1048576) & 0x1FFFFF;
        uint64_t uy = static_cast<uint64_t>(vy + 1048576) & 0x1FFFFF;
        uint64_t uz = static_cast<uint64_t>(vz + 1048576) & 0x1FFFFF;

        keys_out[idx] = (ux << 42) | (uy << 21) | uz;
        acc_out[idx] = {pos.x, pos.y, pos.z, n.x, n.y, n.z, pt.curvature, 1};
    }

    __global__ void finalizeVoxelCentroidsKernel(
        const VoxelAccumulator* acc_in, int unique_count, PointNormal_GPU* output
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= unique_count) return;

        VoxelAccumulator acc = acc_in[idx];
        float inv_count = (acc.count > 0) ? (1.0f / static_cast<float>(acc.count)) : 1.0f;

        PointNormal_GPU pt;
        pt.pos = make_float3(acc.px * inv_count, acc.py * inv_count, acc.pz * inv_count);
        pt.curvature = acc.curvature * inv_count;
        pt.normal_trust = 1.0f;

        float nx = acc.nx;
        float ny = acc.ny;
        float nz = acc.nz;
        float norm_sq = nx * nx + ny * ny + nz * nz;
        if (!isnan(norm_sq) && norm_sq > 1e-6f) {
            float inv_norm = rsqrtf(norm_sq);
            pt.normal = make_float3(nx * inv_norm, ny * inv_norm, nz * inv_norm);
        } else {
            pt.normal = make_float3(0.0f, 0.0f, 1.0f);
        }

        if (isnan(pt.normal.x) || isnan(pt.normal.y) || isnan(pt.normal.z)) {
            pt.normal = make_float3(0.0f, 0.0f, 1.0f);
        }

        output[idx] = pt;
    }

    __global__ void extractVoxelKeysKernel(const PointNormal_GPU* points, size_t num_points, float voxel_size, uint64_t* keys, uint32_t* indices) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) return;

        float3 pos = points[idx].pos;
        int vx = static_cast<int>(floorf(pos.x / voxel_size));
        int vy = static_cast<int>(floorf(pos.y / voxel_size));
        int vz = static_cast<int>(floorf(pos.z / voxel_size));

        uint64_t ux = static_cast<uint64_t>(vx + 1048576) & 0x1FFFFF;
        uint64_t uy = static_cast<uint64_t>(vy + 1048576) & 0x1FFFFF;
        uint64_t uz = static_cast<uint64_t>(vz + 1048576) & 0x1FFFFF;

        keys[idx] = (ux << 42) | (uy << 21) | uz;
        indices[idx] = idx;
    }

} // namespace roller::loopclosure::cuda
