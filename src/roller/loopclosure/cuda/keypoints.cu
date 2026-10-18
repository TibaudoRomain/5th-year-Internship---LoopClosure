#include "roller/loopclosure/cuda/keypoints.cuh"
#include "roller/loopclosure/cuda/descriptors.cuh"
#include "roller/loopclosure/Config.hpp"

#include <algorithm>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/search/kdtree.h>
#include <omp.h>

namespace roller::loopclosure::cuda {

    std::vector<roller::loopclosure::keypoints> compute_pcl_cpu_harris3d(const PointNormal_GPU* normalBuffer, int hostNormalCount, const std::vector<PointNormal_GPU>& host_normals, float voxel_size) {
        std::vector<roller::loopclosure::keypoints> output_keypoints;
        if (host_normals.empty() || hostNormalCount <= 0 || !normalBuffer) return output_keypoints;

        pcl::PointCloud<pcl::PointNormal>::Ptr cloud = importSurfaceCloud(normalBuffer, hostNormalCount, host_normals, voxel_size);
        if (!cloud || cloud->empty()) return output_keypoints;

        pcl::HarrisKeypoint3D<pcl::PointNormal, pcl::PointXYZI, pcl::PointNormal> harris;
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
        harris.setSearchMethod(tree);
        harris.setInputCloud(cloud);
        harris.setNormals(cloud);
        float radius = (Config::estimate_keypoints::harris3d::radius > 0.0) ? static_cast<float>(Config::estimate_keypoints::harris3d::radius) : (voxel_size * 4.0f);
        harris.setRadius(radius);
        harris.setThreshold(static_cast<float>(Config::estimate_keypoints::harris3d::score_threshold));
        harris.setNonMaxSupression(true);
        harris.setNumberOfThreads(omp_get_max_threads());

        pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_pcl(new pcl::PointCloud<pcl::PointXYZI>());
        harris.compute(*keypoints_pcl);

        int count = static_cast<int>(keypoints_pcl->points.size());
        output_keypoints.resize(count);

        pcl::KdTreeFLANN<pcl::PointNormal> kdtree;
        kdtree.setInputCloud(cloud);

        #pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            roller::loopclosure::keypoints kp;
            kp.pos.x = keypoints_pcl->points[i].x;
            kp.pos.y = keypoints_pcl->points[i].y;
            kp.pos.z = keypoints_pcl->points[i].z;
            kp.score = keypoints_pcl->points[i].intensity;

            pcl::PointNormal search_point;
            search_point.x = kp.pos.x;
            search_point.y = kp.pos.y;
            search_point.z = kp.pos.z;

            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);

            if (kdtree.nearestKSearch(search_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                int nn_idx = pointIdxNKNSearch[0];
                kp.normal.x = cloud->points[nn_idx].normal_x;
                kp.normal.y = cloud->points[nn_idx].normal_y;
                kp.normal.z = cloud->points[nn_idx].normal_z;
                kp.normal_trust = 1.0f;
                kp.index_on_map = nn_idx;
            } else {
                kp.normal = make_float3(0.0f, 0.0f, 1.0f);
                kp.normal_trust = 0.0f;
                kp.index_on_map = 0;
            }
            output_keypoints[i] = kp;
        }

        return output_keypoints;
    }

    std::vector<roller::loopclosure::keypoints> compute_pcl_cpu_iss3d(const PointNormal_GPU* normalBuffer, int hostNormalCount, const std::vector<PointNormal_GPU>& host_normals, float voxel_size) {
        std::vector<roller::loopclosure::keypoints> output_keypoints;
        if (host_normals.empty() || hostNormalCount <= 0 || !normalBuffer) return output_keypoints;

        pcl::PointCloud<pcl::PointNormal>::Ptr cloud = importSurfaceCloud(normalBuffer, hostNormalCount, host_normals, voxel_size);
        if (!cloud || cloud->empty()) return output_keypoints;

        pcl::ISSKeypoint3D<pcl::PointNormal, pcl::PointNormal> iss;
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
        iss.setSearchMethod(tree);
        iss.setInputCloud(cloud);

        float search_rad = (Config::estimate_keypoints::iss::searchRadius > 0.0) ? static_cast<float>(Config::estimate_keypoints::iss::searchRadius) : (voxel_size * 6.0f);
        float sal_radius = search_rad;
        float nms_radius = search_rad * 0.75f;

        iss.setSalientRadius(sal_radius);
        iss.setNonMaxRadius(nms_radius);
        iss.setThreshold21(static_cast<float>(Config::estimate_keypoints::iss::gamma_21));
        iss.setThreshold32(static_cast<float>(Config::estimate_keypoints::iss::gamma_32));
        iss.setNumberOfThreads(omp_get_max_threads());

        pcl::PointCloud<pcl::PointNormal>::Ptr keypoints_pcl(new pcl::PointCloud<pcl::PointNormal>());
        iss.compute(*keypoints_pcl);

        int count = static_cast<int>(keypoints_pcl->points.size());
        output_keypoints.resize(count);

        pcl::KdTreeFLANN<pcl::PointNormal> kdtree;
        kdtree.setInputCloud(cloud);

        #pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            roller::loopclosure::keypoints kp;
            kp.pos.x = keypoints_pcl->points[i].x;
            kp.pos.y = keypoints_pcl->points[i].y;
            kp.pos.z = keypoints_pcl->points[i].z;
            kp.score = 1.0f;

            pcl::PointNormal search_point;
            search_point.x = kp.pos.x;
            search_point.y = kp.pos.y;
            search_point.z = kp.pos.z;

            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);

            if (kdtree.nearestKSearch(search_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                int nn_idx = pointIdxNKNSearch[0];
                kp.normal.x = cloud->points[nn_idx].normal_x;
                kp.normal.y = cloud->points[nn_idx].normal_y;
                kp.normal.z = cloud->points[nn_idx].normal_z;
                kp.normal_trust = 1.0f;
                kp.index_on_map = nn_idx;
            } else {
                kp.normal = make_float3(0.0f, 0.0f, 1.0f);
                kp.normal_trust = 0.0f;
                kp.index_on_map = 0;
            }
            output_keypoints[i] = kp;
        }

        return output_keypoints;
    }

    float* compute_harris3d_scores(
        const nanovdb::UInt32Grid* normalMap, const PointNormal_GPU* normalBuffer, int pointCount, float VoxelSize, float SearchRadius,
        float* d_scales, bool scale_aware, int num_scales, float scale_multiplier
    ) {
        if (SearchRadius < VoxelSize) SearchRadius = VoxelSize;

        int threads = 512;
        int blocks = (pointCount + threads - 1) / threads;

        float* d_scores = nullptr;
        cudaMalloc(&d_scores, static_cast<size_t>(pointCount) * sizeof(float));

        float k = static_cast<float>(Config::estimate_keypoints::harris3d_cuda::k);
        Harris3DKernel<<<blocks, threads>>>(
            normalMap, normalBuffer, d_scores, d_scales, pointCount, SearchRadius, VoxelSize, k, scale_aware, num_scales, scale_multiplier
        );
        cudaDeviceSynchronize();
        return d_scores;
    }

    float* compute_iss3d_scores(
        const nanovdb::UInt32Grid* normalMap, const PointNormal_GPU* normalBuffer, int pointCount, float VoxelSize, float SearchRadius,
        float gamma21, float gamma32, float* d_scales, bool scale_aware, int num_scales, float scale_multiplier
    ) {
        if (SearchRadius < VoxelSize) SearchRadius = VoxelSize;

        int threads = 512;
        int blocks = (pointCount + threads - 1) / threads;

        float* d_scores = nullptr;
        cudaMalloc(&d_scores, static_cast<size_t>(pointCount) * sizeof(float));

        ISS3DKernel<<<blocks, threads>>>(
            normalMap, normalBuffer, d_scores, d_scales, pointCount, SearchRadius, VoxelSize, gamma21, gamma32, scale_aware, num_scales, scale_multiplier
        );
        cudaDeviceSynchronize();
        return d_scores;
    }

    std::vector<roller::loopclosure::keypoints> filter_keypoints_number(
        const PointNormal_GPU* normalBuffer, const float* d_scores, const float* d_scales, int Num_Keypoints, int pointCount
    ) {
        std::vector<roller::loopclosure::keypoints> result_keypoints;
        if (pointCount <= 0 || Num_Keypoints <= 0 || !normalBuffer || !d_scores) return result_keypoints;

        int threads = 512;
        int blocks = (pointCount + threads - 1) / threads;

        roller::loopclosure::keypoints* device_keypoints = nullptr;
        int* device_keypoint_count = nullptr;

        cudaMalloc(&device_keypoints, static_cast<size_t>(Num_Keypoints) * sizeof(roller::loopclosure::keypoints));
        cudaMalloc(&device_keypoint_count, sizeof(int));
        cudaMemset(device_keypoint_count, 0, sizeof(int));

        NumberFilterKernel<<<blocks, threads>>>(normalBuffer, d_scores, d_scales, Num_Keypoints, pointCount, device_keypoints, device_keypoint_count);
        cudaDeviceSynchronize();

        int host_keypoint_count = 0;
        cudaMemcpy(&host_keypoint_count, device_keypoint_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (host_keypoint_count > 0) {
            result_keypoints.resize(static_cast<size_t>(host_keypoint_count));
            cudaMemcpy(result_keypoints.data(), device_keypoints, static_cast<size_t>(host_keypoint_count) * sizeof(roller::loopclosure::keypoints), cudaMemcpyDeviceToHost);
        }

        cudaFree(device_keypoint_count);
        cudaFree(device_keypoints);
        return result_keypoints;
    }

    std::vector<roller::loopclosure::keypoints> filter_keypoints_threshold(
        const PointNormal_GPU* normalBuffer, const float* d_scores, const float* d_scales, int Num_Keypoints_Max, int pointCount, float min_score_threshold
    ) {
        std::vector<roller::loopclosure::keypoints> result_keypoints;
        if (pointCount <= 0 || Num_Keypoints_Max <= 0 || !normalBuffer || !d_scores) return result_keypoints;

        int threads = 512;
        int blocks = (pointCount + threads - 1) / threads;

        roller::loopclosure::keypoints* device_keypoints = nullptr;
        int* device_keypoint_count = nullptr;

        cudaMalloc(&device_keypoints, static_cast<size_t>(Num_Keypoints_Max) * sizeof(roller::loopclosure::keypoints));
        cudaMalloc(&device_keypoint_count, sizeof(int));
        cudaMemset(device_keypoint_count, 0, sizeof(int));

        ThresholdFilterKernel<<<blocks, threads>>>(normalBuffer, d_scores, d_scales, Num_Keypoints_Max, pointCount, min_score_threshold, device_keypoints, device_keypoint_count);
        cudaDeviceSynchronize();

        int host_keypoint_count = 0;
        cudaMemcpy(&host_keypoint_count, device_keypoint_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (host_keypoint_count > 0) {
            result_keypoints.resize(static_cast<size_t>(host_keypoint_count));
            cudaMemcpy(result_keypoints.data(), device_keypoints, static_cast<size_t>(host_keypoint_count) * sizeof(roller::loopclosure::keypoints), cudaMemcpyDeviceToHost);
        }

        cudaFree(device_keypoint_count);
        cudaFree(device_keypoints);
        return result_keypoints;
    }

    std::vector<roller::loopclosure::keypoints> filter_keypoints_nms(
        const nanovdb::UInt32Grid* normalMap, const PointNormal_GPU* normalBuffer, const float* d_scores, const float* d_scales,
        int pointCount, float nmsRadius, float voxelSize, float score_threshold, int Num_Keypoints_Max
    ) {
        std::vector<roller::loopclosure::keypoints> host_keypoints;
        if (pointCount <= 0 || Num_Keypoints_Max <= 0 || !normalBuffer || !d_scores || !normalMap) return host_keypoints;

        if (nmsRadius < voxelSize) nmsRadius = voxelSize;

        int threads = 512;
        int blocks = (pointCount + threads - 1) / threads;

        roller::loopclosure::keypoints* device_keypoints = nullptr;
        int* device_keypoint_count = nullptr;

        cudaMalloc(&device_keypoints, static_cast<size_t>(Num_Keypoints_Max) * sizeof(roller::loopclosure::keypoints));
        cudaMalloc(&device_keypoint_count, sizeof(int));
        cudaMemset(device_keypoint_count, 0, sizeof(int));

        NMS3DKernel<<<blocks, threads>>>(
            normalMap, normalBuffer, d_scores, d_scales, pointCount, nmsRadius, voxelSize, 
            score_threshold, Num_Keypoints_Max, device_keypoints, device_keypoint_count
        );
        cudaDeviceSynchronize();

        int host_keypoint_count = 0;
        cudaMemcpy(&host_keypoint_count, device_keypoint_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (host_keypoint_count > 0) {
            host_keypoints.resize(static_cast<size_t>(host_keypoint_count));
            cudaMemcpy(host_keypoints.data(), device_keypoints, static_cast<size_t>(host_keypoint_count) * sizeof(roller::loopclosure::keypoints), cudaMemcpyDeviceToHost);

            std::vector<roller::loopclosure::keypoints> unique_keypoints;
            unique_keypoints.reserve(host_keypoints.size());
            std::set<std::tuple<int, int, int>> seen_voxels;

            for (const auto& kp : host_keypoints) {
                int ix = static_cast<int>(std::round(kp.pos.x * 1000.0f));
                int iy = static_cast<int>(std::round(kp.pos.y * 1000.0f));
                int iz = static_cast<int>(std::round(kp.pos.z * 1000.0f));

                auto key = std::make_tuple(ix, iy, iz);
                if (seen_voxels.find(key) == seen_voxels.end()) {
                    seen_voxels.insert(key);
                    unique_keypoints.push_back(kp);
                }
            }
            host_keypoints = std::move(unique_keypoints);
            host_keypoint_count = static_cast<int>(host_keypoints.size());
        }

        if (device_keypoint_count) cudaFree(device_keypoint_count);
        if (device_keypoints) cudaFree(device_keypoints);

        return host_keypoints;
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr importKeypointCloud(const std::vector<roller::loopclosure::keypoints>& keypoints, float radius, float voxel_size) {
        int keypoint_count = static_cast<int>(keypoints.size());
        
        pcl::PointCloud<pcl::PointNormal>::Ptr keypoint_cloud(new pcl::PointCloud<pcl::PointNormal>());
        keypoint_cloud->points.resize(keypoint_count);
        keypoint_cloud->width = keypoint_count;
        keypoint_cloud->height = 1;
        keypoint_cloud->is_dense = true;
        
        #pragma omp parallel for
        for (int i = 0; i < keypoint_count; ++i) {
            keypoint_cloud->points[i].x = keypoints[i].pos.x;
            keypoint_cloud->points[i].y = keypoints[i].pos.y;
            keypoint_cloud->points[i].z = keypoints[i].pos.z;
            keypoint_cloud->points[i].normal_x = keypoints[i].normal.x;
            keypoint_cloud->points[i].normal_y = keypoints[i].normal.y;
            keypoint_cloud->points[i].normal_z = keypoints[i].normal.z;
        }

        float keypoint_voxel_leaf = voxel_size * Config::estimate_keypoints::keypoint_sampler_size;
        if (keypoint_voxel_leaf <= 0.0f) keypoint_voxel_leaf = radius;

        pcl::PointCloud<pcl::PointNormal>::Ptr keypoint_cloud_downsampled(new pcl::PointCloud<pcl::PointNormal>()); 
        pcl::VoxelGrid<pcl::PointNormal> kp_filter;
        kp_filter.setInputCloud(keypoint_cloud);
        kp_filter.setLeafSize(keypoint_voxel_leaf, keypoint_voxel_leaf, keypoint_voxel_leaf);
        kp_filter.filter(*keypoint_cloud_downsampled);

        return keypoint_cloud_downsampled;
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr keypointsToPointCloud(const std::vector<roller::loopclosure::keypoints>& keypoints) {
        int keypoint_count = static_cast<int>(keypoints.size());
        
        pcl::PointCloud<pcl::PointNormal>::Ptr keypoint_cloud(new pcl::PointCloud<pcl::PointNormal>());
        keypoint_cloud->points.resize(keypoint_count);
        keypoint_cloud->width = keypoint_count;
        keypoint_cloud->height = 1;
        keypoint_cloud->is_dense = true;
        
        #pragma omp parallel for
        for (int i = 0; i < keypoint_count; ++i) {
            keypoint_cloud->points[i].x = keypoints[i].pos.x;
            keypoint_cloud->points[i].y = keypoints[i].pos.y;
            keypoint_cloud->points[i].z = keypoints[i].pos.z;
            keypoint_cloud->points[i].normal_x = keypoints[i].normal.x;
            keypoint_cloud->points[i].normal_y = keypoints[i].normal.y;
            keypoint_cloud->points[i].normal_z = keypoints[i].normal.z;
        }

        return keypoint_cloud;
    }

    void updateKeypointsAfterDownsampling(std::vector<roller::loopclosure::keypoints>& keypoints, const pcl::PointCloud<pcl::PointNormal>::Ptr& keypoint_cloud_downsampled) {
        int new_keypoint_count = keypoint_cloud_downsampled->points.size();

        keypoints.resize(new_keypoint_count);
        
        #pragma omp parallel for
        for (int i = 0; i < new_keypoint_count; ++i) {
            roller::loopclosure::keypoints kp;
            kp.pos.x = keypoint_cloud_downsampled->points[i].x;
            kp.pos.y = keypoint_cloud_downsampled->points[i].y;
            kp.pos.z = keypoint_cloud_downsampled->points[i].z;
            
            kp.normal.x = keypoint_cloud_downsampled->points[i].normal_x;
            kp.normal.y = keypoint_cloud_downsampled->points[i].normal_y;
            kp.normal.z = keypoint_cloud_downsampled->points[i].normal_z;
            
            kp.score = 0.0f; 
            kp.normal_trust = 1.0f;
            
            keypoints[i] = kp; 
        }
    }

} // namespace roller::loopclosure::cuda
