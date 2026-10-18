#include "roller/loopclosure/cuda/descriptors.cuh"
#include "roller/loopclosure/cuda/keypoints.cuh"
#include "roller/loopclosure/cuda/pointcloud.cuh"

#include <chrono>
#include <iostream>
#include <omp.h>

#include <pcl/common/io.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/esf.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/filters/radius_outlier_removal.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>

namespace roller::loopclosure::cuda {

    pcl::PointCloud<pcl::PointNormal>::Ptr importSurfaceCloud(const PointNormal_GPU* normalBuffer, int pointCount, const std::vector<PointNormal_GPU>& host_normals, float voxel_size) {
        if (pointCount <= 0 || !normalBuffer) {
            return std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
        }

        if (voxel_size <= 0.0f) {
            if (host_normals.empty()) return std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
            pcl::PointCloud<pcl::PointNormal>::Ptr surface_cloud(new pcl::PointCloud<pcl::PointNormal>());
            surface_cloud->points.resize(pointCount);
            surface_cloud->width = pointCount;
            surface_cloud->height = 1;
            surface_cloud->is_dense = true;
            #pragma omp parallel for
            for (int i = 0; i < pointCount; ++i) {
                surface_cloud->points[i].x = host_normals[i].pos.x;
                surface_cloud->points[i].y = host_normals[i].pos.y;
                surface_cloud->points[i].z = host_normals[i].pos.z;
                float nx = host_normals[i].normal.x;
                float ny = host_normals[i].normal.y;
                float nz = host_normals[i].normal.z;
                float nlen_sq = nx * nx + ny * ny + nz * nz;
                if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz) || nlen_sq < 1e-6f) {
                    nx = 0.0f; ny = 0.0f; nz = 1.0f;
                } else {
                    float inv_n = 1.0f / std::sqrt(nlen_sq);
                    nx *= inv_n; ny *= inv_n; nz *= inv_n;
                }
                surface_cloud->points[i].normal_x = nx;
                surface_cloud->points[i].normal_y = ny;
                surface_cloud->points[i].normal_z = nz;
                surface_cloud->points[i].curvature = host_normals[i].curvature;
            }
            return surface_cloud;
        }

        PointNormal_GPU* d_downsampled = nullptr;
        cudaMalloc(&d_downsampled, pointCount * sizeof(PointNormal_GPU));

        int unique_count = downsamplePointNormalBufferGPU(normalBuffer, pointCount, d_downsampled, voxel_size);

        std::vector<PointNormal_GPU> h_downsampled(unique_count);
        if (unique_count > 0) {
            cudaMemcpy(h_downsampled.data(), d_downsampled, static_cast<size_t>(unique_count) * sizeof(PointNormal_GPU), cudaMemcpyDeviceToHost);
        }
        cudaFree(d_downsampled);

        pcl::PointCloud<pcl::PointNormal>::Ptr surface_cloud(new pcl::PointCloud<pcl::PointNormal>());
        surface_cloud->points.resize(unique_count);
        surface_cloud->width = unique_count;
        surface_cloud->height = 1;
        surface_cloud->is_dense = true;

        #pragma omp parallel for
        for (int i = 0; i < unique_count; ++i) {
            surface_cloud->points[i].x = h_downsampled[i].pos.x;
            surface_cloud->points[i].y = h_downsampled[i].pos.y;
            surface_cloud->points[i].z = h_downsampled[i].pos.z;
            float nx = h_downsampled[i].normal.x;
            float ny = h_downsampled[i].normal.y;
            float nz = h_downsampled[i].normal.z;
            float nlen_sq = nx * nx + ny * ny + nz * nz;
            if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz) || nlen_sq < 1e-6f) {
                nx = 0.0f; ny = 0.0f; nz = 1.0f;
            } else {
                float inv_n = 1.0f / std::sqrt(nlen_sq);
                nx *= inv_n; ny *= inv_n; nz *= inv_n;
            }
            surface_cloud->points[i].normal_x = nx;
            surface_cloud->points[i].normal_y = ny;
            surface_cloud->points[i].normal_z = nz;
            surface_cloud->points[i].curvature = h_downsampled[i].curvature;
        }

        // Apply Radius Outlier Removal to eliminate detached floating point islands behind the surface
        if (surface_cloud->size() > 10) {
            pcl::RadiusOutlierRemoval<pcl::PointNormal> ror;
            ror.setInputCloud(surface_cloud);
            ror.setRadiusSearch(0.35f);
            ror.setMinNeighborsInRadius(6);
            pcl::PointCloud<pcl::PointNormal>::Ptr cleaned_cloud(new pcl::PointCloud<pcl::PointNormal>());
            ror.filter(*cleaned_cloud);
            return cleaned_cloud;
        }

        return surface_cloud;
    }

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFH(std::vector<keypoints>& keypoints, const PointNormal_GPU* normalBuffer, int pointCount, const std::vector<PointNormal_GPU>& host_normals, int k, float radius, float voxel_size, bool use_k){
        if (keypoints.empty() || pointCount == 0) {return {};}

        std::cout << "[computeFPFH] Starting FPFH calculation for " << keypoints.size() << " keypoints..." << std::endl;
        auto fpfh_out = std::make_shared<pcl::PointCloud<pcl::FPFHSignature33>>();
        
        // Downsample search surface to descriptor resolution (0.15m) using CUDA GPU centroid reduction
        float desc_surf_voxel = std::max(voxel_size * 3.0f, 0.15f);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto surface_cloud_downsampled = importSurfaceCloud(normalBuffer, pointCount, host_normals, desc_surf_voxel);
        auto t1 = std::chrono::high_resolution_clock::now();
        double import_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "[computeFPFH] GPU search surface downsampled to voxel_size = " << desc_surf_voxel 
                  << " m (" << surface_cloud_downsampled->size() << " surface points, created in " << import_time << " ms)" << std::endl;

        auto keypoint_cloud = keypointsToPointCloud(keypoints);
        
        pcl::FPFHEstimationOMP<pcl::PointNormal, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
        int num_threads = omp_get_max_threads();
        fpfh.setNumberOfThreads(num_threads); 
        std::cout << "[computeFPFH] Using " << num_threads << " OpenMP threads" << std::endl;
        
        fpfh.setInputCloud(keypoint_cloud);
        fpfh.setSearchSurface(surface_cloud_downsampled);
        fpfh.setInputNormals(surface_cloud_downsampled);

        pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
        fpfh.setSearchMethod(tree);
    
        if (use_k) {
            std::cout << "[computeFPFH] Setting KSearch = " << k << std::endl;
            fpfh.setKSearch(k);
        } else {
            std::cout << "[computeFPFH] Setting RadiusSearch = " << radius << " m" << std::endl;
            fpfh.setRadiusSearch(radius);
        }
        
        std::cout << "[computeFPFH] Running fpfh.compute()..." << std::endl;
        auto t2 = std::chrono::high_resolution_clock::now();
        fpfh.compute(*fpfh_out);
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "[computeFPFH] fpfh.compute() completed in " << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms" << std::endl;

        return fpfh_out;
    }

    pcl::PointCloud<pcl::SHOT352>::Ptr computeSHOT(std::vector<keypoints>& keypoints, const PointNormal_GPU* normalBuffer, int pointCount, const std::vector<PointNormal_GPU>& host_normals, float radius, float voxel_size){
        if (keypoints.empty() || pointCount == 0 || host_normals.empty()) {return {};}

        auto shot_out = std::make_shared<pcl::PointCloud<pcl::SHOT352>>();
        
        auto surface_cloud_downsampled = importSurfaceCloud(normalBuffer, pointCount, host_normals, voxel_size);
        auto keypoint_cloud = keypointsToPointCloud(keypoints);

        pcl::SHOTEstimationOMP<pcl::PointNormal, pcl::PointNormal, pcl::SHOT352> shot;
        shot.setNumberOfThreads(omp_get_max_threads()); 
        
        shot.setInputCloud(keypoint_cloud);
        shot.setSearchSurface(surface_cloud_downsampled);
        shot.setInputNormals(surface_cloud_downsampled);

        pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
        shot.setSearchMethod(tree);
        
        shot.setRadiusSearch(radius);
        
        shot.compute(*shot_out);

        return shot_out;
    }

    pcl::PointCloud<HPVNDSignature324>::Ptr computeHPNVD(std::vector<keypoints>& keypoints, const PointNormal_GPU* normalBuffer, int pointCount, const std::vector<PointNormal_GPU>& host_normals, float radius, int bin_num, float voxel_size) {
        auto hpnvd_out = std::make_shared<pcl::PointCloud<HPVNDSignature324>>();
        if (keypoints.empty()) return hpnvd_out;

        pcl::PointCloud<pcl::PointNormal>::Ptr keypoint_cloud = keypointsToPointCloud(keypoints);
        pcl::PointCloud<pcl::PointNormal>::Ptr surface_cloud_downsampled = importSurfaceCloud(normalBuffer, pointCount, host_normals, voxel_size);

        pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_surface(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*surface_cloud_downsampled, *xyz_surface);

        std::vector<int> keypoint_indices_in_surface;
        keypoint_indices_in_surface.reserve(keypoint_cloud->size());

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(xyz_surface);

        for (size_t i = 0; i < keypoint_cloud->size(); ++i) {
            pcl::PointXYZ kp(keypoint_cloud->points[i].x, keypoint_cloud->points[i].y, keypoint_cloud->points[i].z);
            std::vector<int> idx(1);
            std::vector<float> dist(1);
            if (kdtree.nearestKSearch(kp, 1, idx, dist) > 0) {
                keypoint_indices_in_surface.push_back(idx[0]);
            }
        }

        HPVND_compute(xyz_surface, keypoint_indices_in_surface, radius, bin_num, hpnvd_out);
        return hpnvd_out;
    }

    pcl::PointCloud<pcl::ESFSignature640>::Ptr computeESF(const PointNormal_GPU* normalBuffer, int pointCount, const std::vector<PointNormal_GPU>& host_normals, float voxel_size) {
        auto esf_out = std::make_shared<pcl::PointCloud<pcl::ESFSignature640>>();
        if (pointCount <= 0 || host_normals.empty()) return esf_out;

        pcl::PointCloud<pcl::PointNormal>::Ptr surface_cloud_downsampled = importSurfaceCloud(normalBuffer, pointCount, host_normals, voxel_size);
        if (surface_cloud_downsampled->empty()) return esf_out;

        pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_surface(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*surface_cloud_downsampled, *xyz_surface);

        pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
        esf.setInputCloud(xyz_surface);
        esf.compute(*esf_out);

        return esf_out;
    }

    pcl::PointCloud<pcl::ESFSignature640>::Ptr computeLeafESFDescriptorsGPU(
        const float3* d_surface,
        const FilledLeaf* d_filled_leafs,
        const int* d_point_indices_by_leaf,
        int leaf_count,
        int total_leaf_points
    ) {
        auto esf_out = std::make_shared<pcl::PointCloud<pcl::ESFSignature640>>();
        if (leaf_count <= 0 || !d_surface || !d_filled_leafs || !d_point_indices_by_leaf || total_leaf_points <= 0) return esf_out;

        std::vector<FilledLeaf> h_leafs(leaf_count);
        cudaMemcpy(h_leafs.data(), d_filled_leafs, leaf_count * sizeof(FilledLeaf), cudaMemcpyDeviceToHost);

        std::vector<int> h_point_indices_by_leaf(total_leaf_points);
        cudaMemcpy(h_point_indices_by_leaf.data(), d_point_indices_by_leaf, total_leaf_points * sizeof(int), cudaMemcpyDeviceToHost);

        int max_idx = -1;
        for (int idx : h_point_indices_by_leaf) {
            if (idx > max_idx) max_idx = idx;
        }
        int host_surf_count = max_idx + 1;
        if (host_surf_count <= 0) return esf_out;

        std::vector<float3> h_surface(host_surf_count);
        cudaMemcpy(h_surface.data(), d_surface, host_surf_count * sizeof(float3), cudaMemcpyDeviceToHost);

        esf_out->resize(leaf_count);

        #pragma omp parallel for
        for (int j = 0; j < leaf_count; ++j) {
            const FilledLeaf& leaf = h_leafs[j];
            if (leaf.count < 10 || leaf.offset < 0 || (leaf.offset + leaf.count) > total_leaf_points) {
                pcl::ESFSignature640 zero_sig{};
                esf_out->points[j] = zero_sig;
                continue;
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr leaf_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            leaf_cloud->points.resize(leaf.count);

            for (int i = 0; i < leaf.count; ++i) {
                int pt_idx = h_point_indices_by_leaf[leaf.offset + i];
                if (pt_idx >= 0 && pt_idx < host_surf_count) {
                    float3 pt = h_surface[pt_idx];
                    leaf_cloud->points[i] = pcl::PointXYZ(pt.x, pt.y, pt.z);
                }
            }

            pcl::PointCloud<pcl::ESFSignature640> single_esf;
            pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
            esf.setInputCloud(leaf_cloud);
            esf.compute(single_esf);

            if (!single_esf.empty()) {
                esf_out->points[j] = single_esf.points[0];
            } else {
                pcl::ESFSignature640 zero_sig{};
                esf_out->points[j] = zero_sig;
            }
        }

        return esf_out;
    }

} // namespace roller::loopclosure::cuda
