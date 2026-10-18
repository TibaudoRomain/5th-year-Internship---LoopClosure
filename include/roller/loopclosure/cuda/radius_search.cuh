#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <roller/loopclosure/cuda/types.hpp>

namespace roller::loopclosure::cuda {

    /**
     * @brief CUDA kernel to count number of surface points within radius for each query point.
     */
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
    );

    /**
     * @brief CUDA kernel to fill the contiguous patch indices buffer for each query point.
     */
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
    );

    /**
     * @brief Host function to build spatial hash grid on GPU and perform unlimited radius search.
     * 
     * @param d_surface Device buffer for surface points (float3*)
     * @param surface_count Number of surface points
     * @param d_query Device buffer for query points (float3*)
     * @param query_count Number of query points
     * @param radius Radius for spatial neighbor search
     * @param d_patches_out Output device buffer of struct Patch (allocated for query_count)
     * @param d_patch_indices_out Output device pointer receiving dynamically allocated patch point indices
     * @param total_patch_points_out Output reference receiving total count of indices across all patches
     * @param stream Optional CUDA stream
     */
    void performRadiusSearchGPU(
        const float3* d_surface,
        int surface_count,
        const float3* d_query,
        int query_count,
        float radius,
        Patch* d_patches_out,
        int** d_patch_indices_out,
        int& total_patch_points_out,
        cudaStream_t stream = 0
    );

} // namespace roller::loopclosure::cuda
