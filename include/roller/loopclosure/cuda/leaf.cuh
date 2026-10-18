#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <roller/loopclosure/cuda/types.hpp>

namespace roller::loopclosure::cuda {

    /**
     * @brief CUDA kernel to count raw surface points belonging to each Octree Leaf box using spatial hashing.
     */
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
    );

    /**
     * @brief CUDA kernel to fill matching raw surface point indices into each Octree Leaf box using spatial hashing.
     */
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
    );

    /**
     * @brief Attributes raw surface points to Octree Leaf nodes on GPU using Spatial Hashing.
     */
    void attributePointsToLeafsGPU(
        const float3* d_surface,
        int surface_count,
        const LeafNodeInfo* d_leafs,
        int leaf_count,
        FilledLeaf* d_filled_leafs_out,
        int** d_point_indices_by_leaf_out,
        int& total_leaf_points_out,
        cudaStream_t stream = 0
    );

} // namespace roller::loopclosure::cuda
