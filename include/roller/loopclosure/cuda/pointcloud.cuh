#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#undef NANOVDB_USE_OPENVDB
#include <nanovdb/NanoVDB.h>
#include <roller/loopclosure/cuda/types.hpp>

namespace roller::loopclosure::cuda {

    /**
     * @brief CUDA kernel to convert NanoVDB TSDF FloatGrid surface voxels into a float3 3D position buffer.
     */
    __global__ void convertTSDFToFloat3Kernel(
        const nanovdb::NanoGrid<float>* sdf_grid,
        float3* d_points_out,
        int* d_point_counter,
        float max_distance_to_tsdf,
        float voxel_size,
        nanovdb::Coord bbox_min,
        int dimX, int dimY, int dimZ
    );

    /**
     * @brief Host wrapper function to extract 3D surface points from NanoVDB TSDF FloatGrid directly into a float3 device buffer.
     */
    int convertTSDFToFloat3GPU(
        const nanovdb::NanoGrid<float>* d_sdf_grid,
        const nanovdb::CoordBBox& bbox,
        float voxel_size,
        float max_distance_to_tsdf,
        float3* d_points_out,
        int* d_point_counter,
        cudaStream_t stream = 0
    );

    /**
     * @brief CUDA kernel to extract PointNormal_GPU surface points from NanoVDB TSDF FloatGrid.
     */
    struct VoxelAccumulator {
        float px, py, pz;
        float nx, ny, nz;
        float curvature;
        int count;

        __host__ __device__ VoxelAccumulator operator+(const VoxelAccumulator& rhs) const {
            return {px + rhs.px, py + rhs.py, pz + rhs.pz,
                    nx + rhs.nx, ny + rhs.ny, nz + rhs.nz,
                    curvature + rhs.curvature,
                    count + rhs.count};
        }
    };

    __global__ void extractTSDFSurfaceKernel(
        const nanovdb::NanoGrid<float>* sdf_grid,
        PointNormal_GPU* dest,
        int* normalCounter,
        float max_distance_to_tsdf,
        float voxelSize,
        nanovdb::Coord bbox_min,
        int dimX, int dimY, int dimZ
    );

    /**
     * @brief Downsamples a device PointNormal_GPU buffer using GPU voxel grid centroid reduction.
     */
    int downsamplePointNormalBufferGPU(
        const PointNormal_GPU* d_input,
        int input_count,
        PointNormal_GPU* d_output,
        float downsample_voxel_size
    );

    __global__ void extractMonotonicVoxelKeysKernel(
        const PointNormal_GPU* points, int num_points, float voxel_size,
        uint64_t* keys_out, VoxelAccumulator* acc_out
    );

    __global__ void finalizeVoxelCentroidsKernel(
        const VoxelAccumulator* acc_in, int unique_count, PointNormal_GPU* output
    );

    __global__ void extractVoxelKeysKernel(const PointNormal_GPU* points, size_t num_points, float voxel_size, uint64_t* keys, uint32_t* indices);

} // namespace roller::loopclosure::cuda
