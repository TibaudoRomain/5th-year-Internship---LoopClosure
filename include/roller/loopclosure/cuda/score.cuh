#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#undef NANOVDB_USE_OPENVDB
#include <nanovdb/NanoVDB.h>
#include <roller/loopclosure/cuda/types.hpp>

namespace roller::loopclosure::cuda {

    /**
     * @brief CUDA kernel to compute spatial 3x3 covariance matrix and centroid for each patch.
     * 
     * @param d_surface Device array of surface points (float3*)
     * @param surface_count Total number of surface points
     * @param d_patches Device array of Patch structures
     * @param d_patch_indices Device array of contiguous patch indices
     * @param query_count Total number of patches / query points
     * @param d_covariances_out Device output buffer for PatchCovariance (allocated for query_count)
     */
    __global__ void computePatchCovariancesKernel(
        const float3* d_surface,
        int surface_count,
        const Patch* d_patches,
        const int* d_patch_indices,
        int query_count,
        PatchCovariance* d_covariances_out
    );

    /**
     * @brief Host wrapper function to compute 3x3 covariance matrices for all patches on GPU.
     * 
     * @param d_surface Device array of surface points (float3*)
     * @param surface_count Total number of surface points
     * @param d_patches Device array of Patch structures
     * @param d_patch_indices Device array of contiguous patch indices
     * @param query_count Total number of patches / query points
     * @param d_covariances_out Device output buffer for PatchCovariance (allocated for query_count)
     * @param stream Optional CUDA stream
     */
    void computePatchCovariancesGPU(
        const float3* d_surface,
        int surface_count,
        const Patch* d_patches,
        const int* d_patch_indices,
        int query_count,
        PatchCovariance* d_covariances_out,
        cudaStream_t stream = 0
    );

    /**
     * @brief CUDA kernel to compute patch score = 3.0 * e3 / (e1 + e2 + e3) from 3x3 covariance matrix eigenvalues.
     */
    __global__ void computePatchScoresKernel(
        const PatchCovariance* d_covariances,
        int query_count,
        float* d_scores_out
    );

    /**
     * @brief Host wrapper function to compute patch scores score = 3.0 * e3 / (e1 + e2 + e3) on GPU.
     * 
     * @param d_covariances Device buffer of PatchCovariance structures
     * @param query_count Total number of patches / query points
     * @param d_scores_out Device output buffer for scores (allocated for query_count)
     * @param stream Optional CUDA stream
     */
    void computePatchScoresGPU(
        const PatchCovariance* d_covariances,
        int query_count,
        float* d_scores_out,
        cudaStream_t stream = 0
    );

} // namespace roller::loopclosure::cuda
