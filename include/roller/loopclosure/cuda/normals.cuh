#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#undef NANOVDB_USE_OPENVDB
#include <nanovdb/NanoVDB.h>
#include <roller/loopclosure/cuda/types.hpp>

namespace roller::loopclosure::cuda {

    template <int AXIS>
    __global__ void GaussianFilterKernel(const nanovdb::NanoGrid<float>* deviceGrid, float* dest, int dimX, int dimY, int dimZ, nanovdb::Coord bbox_origin, int width);

    __global__ void normalCalcsKernel(const float4* submap, const int* submap_offsets, int nb_of_keyframe, int num_points, PointNormal_GPU* dest, const nanovdb::NanoGrid<float>* sdf_grid, const float3* sensor_xyz, int* normalCounter, float dist_to_sdf, float voxelSize);
    __global__ void curvCalcsKernel(PointNormal_GPU* input_normals, const nanovdb::UInt32Grid* normalsSpatialMap, int total_surface_points, float voxelSize);

    void GaussianFilter(nanovdb::FloatGrid* gpu_Grid_Ptr, nanovdb::CoordBBox bbox, int width, int repetitions);
    void NormalAcquisition(PointNormal_GPU* dest, nanovdb::FloatGrid* Grid, nanovdb::CoordBBox bbox, float dist_to_sdf, float VoxelSize, int* normalCounter);
    void CurvAcquisition(PointNormal_GPU* inp_dest, const nanovdb::UInt32Grid* normalMap, int normalCount, float voxelSize);

} // namespace roller::loopclosure::cuda
