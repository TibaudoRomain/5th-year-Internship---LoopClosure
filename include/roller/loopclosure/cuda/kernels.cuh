#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#undef NANOVDB_USE_OPENVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/util/Ray.h>
#include <roller/loopclosure/cuda/types.hpp>

namespace roller::loopclosure::cuda {

    constexpr int AXIS_X = 0;
    constexpr int AXIS_Y = 1;
    constexpr int AXIS_Z = 2;

    __global__ void extractPositionsKernel(const PointNormal_GPU* input, nanovdb::Vec3f* output, int count);
    __global__ void reinjectToNano(float* inputBuffer, nanovdb::NanoGrid<float>* deviceGrid, int dimX, int dimY, int dimZ, nanovdb::Coord bbox_origin);

    __device__ nanovdb::math::Ray<float> getRay(const int3 &start, const float &direction, float distance_point_sensor, float range_around_point, float voxelSize);
    __device__ nanovdb::math::Ray<float> getRay(const int3 &start, const int3 &direction, float voxelSize, float range_around_point = 0.1f, float sonar_min_range = 0.2f, float sonar_max_range = 15.0f);
    __device__ int3 to_index(const float3 point, float voxelSize);

    template <bool squared>
    __device__ float len(const int3 vector);
    __device__ float3 normalize(const int3 vector);
    __device__ int whichKeyframe(const int global_index, const int* submap_offsets, int nb_of_keyframe);
    __device__ float3 computeNormal(nanovdb::FloatGrid::AccessorType& acc, const nanovdb::Coord& ijk, float voxel_size, bool& is_valid);
    __device__ uint64_t packCoordToKey(int x, int y, int z);
    __device__ void computeEigenvalues3x3(float m11, float m12, float m13, float m22, float m23, float m33, float &l1, float &l2, float &l3);

} // namespace roller::loopclosure::cuda
