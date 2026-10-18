#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot.h>
#include <pcl/features/esf.h>
#include <roller/loopclosure/cuda/types.hpp>

namespace roller::loopclosure::cuda {

    pcl::PointCloud<pcl::PointNormal>::Ptr importSurfaceCloud(
        const PointNormal_GPU* normalBuffer,
        int pointCount,
        const std::vector<PointNormal_GPU>& host_normals,
        float voxel_size
    );

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFH(
        std::vector<keypoints>& keypoints,
        const PointNormal_GPU* normalBuffer,
        int pointCount,
        const std::vector<PointNormal_GPU>& host_normals,
        int k,
        float radius,
        float voxel_size,
        bool use_k
    );

    pcl::PointCloud<pcl::SHOT352>::Ptr computeSHOT(
        std::vector<keypoints>& keypoints,
        const PointNormal_GPU* normalBuffer,
        int pointCount,
        const std::vector<PointNormal_GPU>& host_normals,
        float radius,
        float voxel_size
    );

    pcl::PointCloud<HPVNDSignature324>::Ptr computeHPNVD(
        std::vector<keypoints>& keypoints,
        const PointNormal_GPU* normalBuffer,
        int pointCount,
        const std::vector<PointNormal_GPU>& host_normals,
        float radius,
        int bin_num,
        float voxel_size
    );

    pcl::PointCloud<pcl::ESFSignature640>::Ptr computeESF(
        const PointNormal_GPU* normalBuffer,
        int pointCount,
        const std::vector<PointNormal_GPU>& host_normals,
        float voxel_size
    );

    pcl::PointCloud<pcl::ESFSignature640>::Ptr computeLeafESFDescriptorsGPU(
        const float3* d_surface,
        const FilledLeaf* d_filled_leafs,
        const int* d_point_indices_by_leaf,
        int leaf_count,
        int total_leaf_points
    );

} // namespace roller::loopclosure::cuda
