#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#undef NANOVDB_USE_OPENVDB
#include <nanovdb/NanoVDB.h>
#include <roller/loopclosure/cuda/types.hpp>

namespace roller::loopclosure::cuda {

    __global__ void Harris3DKernel(const nanovdb::UInt32Grid* grid, const PointNormal_GPU* input, float* d_harrisScores, float* d_scales, int pointCount, float searchRadius, float voxelSize, float k = 0.04f, bool scale_aware = false, int num_scales = 1, float scale_multiplier = 1.0f);
    __global__ void ISS3DKernel(const nanovdb::UInt32Grid* grid, const PointNormal_GPU* input, float* d_issScores, float* d_scales, int pointCount, float searchRadius, float voxelSize, float gamma_21, float gamma_32, bool scale_aware = false, int num_scales = 1, float scale_multiplier = 1.0f);
    __global__ void ThresholdFilterKernel(const PointNormal_GPU* input, const float* d_scores, const float* d_scales, int Num_Keypoints_Max, int pointCount, float min_score_threshold, keypoints* d_outputKeypoints, int* d_keypointCount);
    __global__ void NumberFilterKernel(const PointNormal_GPU* input, const float* d_scores, const float* d_scales, int Num_Keypoints, int pointCount, keypoints* d_outputKeypoints,  int* d_keypointCount);
    __global__ void NMS3DKernel(const nanovdb::UInt32Grid* grid, const PointNormal_GPU* input, const float* d_scores, const float* d_scales, int pointCount, float nmsRadius, float voxelSize, float score_threshold, int Num_Keypoints_Max, keypoints* d_outputKeypoints, int* d_keypointCount);

    std::vector<keypoints> compute_pcl_cpu_harris3d(const PointNormal_GPU* normalBuffer, int hostNormalCount, const std::vector<PointNormal_GPU>& host_normals, float voxel_size);
    std::vector<keypoints> compute_pcl_cpu_iss3d(const PointNormal_GPU* normalBuffer, int hostNormalCount, const std::vector<PointNormal_GPU>& host_normals, float voxel_size);
    float* compute_harris3d_scores(const nanovdb::UInt32Grid* normalMap, const PointNormal_GPU* normalBuffer, int pointCount, float VoxelSize, float SearchRadius, float* d_scales = nullptr, bool scale_aware = false, int num_scales = 1, float scale_multiplier = 1.0f);
    float* compute_iss3d_scores(const nanovdb::UInt32Grid* normalMap, const PointNormal_GPU* normalBuffer, int pointCount, float VoxelSize, float SearchRadius, float gamma21, float gamma32, float* d_scales = nullptr, bool scale_aware = false, int num_scales = 1, float scale_multiplier = 1.0f);
    std::vector<keypoints> filter_keypoints_number(const PointNormal_GPU* normalBuffer, const float* d_scores, const float* d_scales, int Num_Keypoints, int pointCount);
    std::vector<keypoints> filter_keypoints_threshold(const PointNormal_GPU* normalBuffer, const float* d_scores, const float* d_scales, int Num_Keypoints_Max, int pointCount, float min_score_threshold);
    std::vector<keypoints> filter_keypoints_nms(const nanovdb::UInt32Grid* normalMap, const PointNormal_GPU* normalBuffer, const float* d_scores, const float* d_scales, int pointCount, float nmsRadius, float voxelSize, float score_threshold, int Num_Keypoints_Max);

    pcl::PointCloud<pcl::PointNormal>::Ptr importKeypointCloud(const std::vector<keypoints>& keypoints, float radius, float voxel_size);
    pcl::PointCloud<pcl::PointNormal>::Ptr keypointsToPointCloud(const std::vector<keypoints>& keypoints);
    void updateKeypointsAfterDownsampling(std::vector<keypoints>& keypoints, const pcl::PointCloud<pcl::PointNormal>::Ptr& keypoint_cloud_downsampled);

} // namespace roller::loopclosure::cuda
