#include "roller/loopclosure/cuda/device_pipeline.cuh"
#include "roller/loopclosure/cuda/device_pipeline.hpp"
#include "roller/loopclosure/cuda/kernels.cuh"

#include <chrono>
#include <iostream>

#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/util/HDDA.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>

namespace roller::loopclosure {

    CudaCore_Wrapper::CudaCore_Wrapper(float voxel_size) 
        : impl_(std::make_unique<CudaCore>(voxel_size)) {}

    CudaCore_Wrapper::~CudaCore_Wrapper() = default;

    CudaResult CudaCore_Wrapper::LDPipeline_device(const Submap& submap, const std::vector<Keyframe>& submap_keyframes, const nanovdb::GridHandle<nanovdb::HostBuffer>& sdf) {
        if (!impl_) {
            std::cerr << "[CudaCore_Wrapper] : Internal implementation is null!" << std::endl;
            return CudaResult();
        }
        return impl_->LDPipeline_device(submap, submap_keyframes, sdf);
    }

    CudaCore::CudaCore(float voxel_size)
        : _VoxelSize(voxel_size),
          gpu_Submap_Ptr(nullptr),
          gpu_SensorPos_Ptr(nullptr),
          gpu_SubmapOffsets_Ptr(nullptr),
          gpu_SDFHandle_Ptr(nullptr),
          normalbuffer(nullptr),
          normalCounter(nullptr),
          gpu_SDF_Ptr(nullptr)
    {
        std::cout << "[CudaCore] : Initialized" << std::endl;
    }
    
    CudaCore::~CudaCore() {
        if (gpu_SDF_Ptr) cudaFree(gpu_SDF_Ptr);
        if (gpu_SDFHandle_Ptr) cudaFree(gpu_SDFHandle_Ptr);
        if (gpu_SubmapOffsets_Ptr) cudaFree(gpu_SubmapOffsets_Ptr);
        if (gpu_SensorPos_Ptr) cudaFree(gpu_SensorPos_Ptr);
        if (gpu_Submap_Ptr) cudaFree(gpu_Submap_Ptr);

        gpu_Submap_Ptr = nullptr;
        gpu_SubmapOffsets_Ptr = nullptr;
        gpu_SensorPos_Ptr = nullptr;
        gpu_SDFHandle_Ptr = nullptr;
        normalbuffer = nullptr;
        normalCounter = nullptr;
        gpu_SDF_Ptr = nullptr;
        
        std::cout << "[CudaCore] : Destroyed" << std::endl;
    }
    
    CudaResult CudaCore::LDPipeline_device(const Submap& submap, const std::vector<Keyframe>& submap_keyframes, const nanovdb::GridHandle<nanovdb::HostBuffer>& sdf){
        std::cout << "[CudaCore] : Starting submap processing" << std::endl;
        cached_surface_cloud_ = nullptr;

        if (submap_keyframes.empty()) {
            std::cout << "[CudaCore] : Empty Submap. Bypassing." << std::endl;
            return this->empty_result();
        }
        size_t active_voxels = sdf.grid<float>()->activeVoxelCount();
        if (active_voxels == 0) {
            std::cout << "[CudaCore] : Empty TSDF Grid. Bypassing." << std::endl;
            return this->empty_result();
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        size_t SDFHandle_Size = sdf.bufferSize();
        cudaMalloc(&gpu_SDFHandle_Ptr, SDFHandle_Size);
        cudaMemcpy(gpu_SDFHandle_Ptr, sdf.data(), SDFHandle_Size, cudaMemcpyHostToDevice);

        gpu_SDF_Ptr = reinterpret_cast<nanovdb::FloatGrid*>(gpu_SDFHandle_Ptr);
        
        std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
        std::cout << "[CudaCore] : TSDF uploaded to GPU in " << duration.count() << " ms" << std::endl;

        if(Config::normal_calculation::gaussian_filter){
            start = std::chrono::high_resolution_clock::now();
            cuda::GaussianFilter(gpu_SDF_Ptr, sdf.grid<float>()->indexBBox(), 
                                Config::normal_calculation::gaussian_width,
                                Config::normal_calculation::gaussian_repetitions);
            duration = std::chrono::high_resolution_clock::now() - start;
            std::cout << "[CudaCore] : TSDF Gaussian filter finished in " << duration.count() << " ms" << std::endl;
        }
        start = std::chrono::high_resolution_clock::now();
        cudaMalloc(&normalbuffer, active_voxels * sizeof(PointNormal_GPU));

        cudaMalloc(&normalCounter, sizeof(int));
        cudaMemset(normalCounter, 0, sizeof(int));

        cuda::NormalAcquisition(normalbuffer, gpu_SDF_Ptr, sdf.grid<float>()->indexBBox(), static_cast<float>(Config::normal_calculation::distance_to_tsdf), _VoxelSize, normalCounter);
        cudaDeviceSynchronize();

        int hostNormalCount = 0;
        cudaMemcpy(&hostNormalCount, normalCounter, sizeof(int), cudaMemcpyDeviceToHost);

        auto normalGridHandle = this->buildGrid(normalbuffer, hostNormalCount, _VoxelSize);
        auto normalMappingGrid = normalGridHandle.deviceGrid<uint32_t>();

        duration = std::chrono::high_resolution_clock::now() - start;
        std::cout << "[CudaCore] : Normal Calculations finished in " << duration.count() << " ms" << std::endl;
        std::vector<PointNormal_GPU> host_normals;
        bool need_host_normals = Config::debug::allow_pointcloud_extraction ||
                                 Config::estimate_keypoints::method == "harris3d_pcl_cpu" ||
                                 Config::estimate_keypoints::method == "iss3d_pcl_cpu" ||
                                 !Config::extract_feature_descriptors::method.empty();
        if (need_host_normals && hostNormalCount > 0) {
            host_normals.resize(hostNormalCount);
            cudaMemcpy(host_normals.data(), normalbuffer, static_cast<size_t>(hostNormalCount) * sizeof(PointNormal_GPU), cudaMemcpyDeviceToHost);
        }
        
        start = std::chrono::high_resolution_clock::now();
        std::vector<float> kp_scores;
        std::vector<keypoints> keypoints = this->keypoint_estimation_step(normalMappingGrid, normalbuffer, hostNormalCount, host_normals, kp_scores);
        
        duration = std::chrono::high_resolution_clock::now() - start;
        std::cout << "[CudaCore] : Keypoints Estimation finished in " << duration.count() << " ms" << std::endl;
        std::cout << "[CudaCore] : Keypoints found = " << keypoints.size() << std::endl;

        CudaResult R = this->computeDescriptors(keypoints, normalbuffer, hostNormalCount, host_normals, kp_scores);
                
        if (gpu_SDF_Ptr) {gpu_SDF_Ptr = nullptr;}

        if (normalCounter) {cudaFree(normalCounter); normalCounter = nullptr;}
        if (normalbuffer) {cudaFree(normalbuffer); normalbuffer = nullptr;}
        if (gpu_SDFHandle_Ptr) {cudaFree(gpu_SDFHandle_Ptr); gpu_SDFHandle_Ptr = nullptr;}
        if (gpu_SensorPos_Ptr) {cudaFree(gpu_SensorPos_Ptr); gpu_SensorPos_Ptr = nullptr;}
        if (gpu_SubmapOffsets_Ptr) {cudaFree(gpu_SubmapOffsets_Ptr); gpu_SubmapOffsets_Ptr = nullptr;}
        if (gpu_Submap_Ptr) {cudaFree(gpu_Submap_Ptr); gpu_Submap_Ptr = nullptr;}
        
        gpu_SDF_Ptr = nullptr;
        std::cout << "[CudaCore] : Core cleaned " << std::endl;

        return R;
    }

    CudaResult CudaCore::empty_result(){
        CudaResult result;
        result.normals = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
        result.keypoints = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        result.descriptors = std::make_shared<pcl::PointCloud<pcl::FPFHSignature33>>(); 
        return result;
    }

    CudaResult CudaCore::results(const std::vector<PointNormal_GPU>& host_normals,
        const std::vector<keypoints>& keypoints, bool allow_pointcloud_extraction, const std::vector<float>& kp_scores
    ){
        CudaResult result;
        size_t keypointCount = keypoints.size();

        result.normals = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
        result.keypoints = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        result.keypoints->resize(keypointCount);

        if (allow_pointcloud_extraction && !host_normals.empty()) {
            size_t normalCount = host_normals.size();
            result.normals->resize(normalCount);

            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(normalCount); ++i) {
                pcl::PointNormal& point = result.normals->operator[](i); 
                const auto& host_n = host_normals[i];

                point.x = host_n.pos.x;
                point.y = host_n.pos.y;
                point.z = host_n.pos.z;
                point.normal_x = host_n.normal.x;
                point.normal_y = host_n.normal.y;
                point.normal_z = host_n.normal.z;
                point.curvature = host_n.curvature;
            }

            result.debug_harrisscore = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
            result.debug_harrisscore->resize(normalCount);

            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(normalCount); ++i) {
                pcl::PointXYZI& point = result.debug_harrisscore->operator[](i);
                const auto& host_n = host_normals[i];

                point.x = host_n.pos.x;
                point.y = host_n.pos.y;
                point.z = host_n.pos.z;
                point.intensity = (i < static_cast<int>(kp_scores.size())) ? kp_scores[i] : 0.0f;
            }
        }

        if(!keypoints.empty()){
            #pragma omp parallel for
            for (size_t i = 0; i < keypointCount; ++i) {
                pcl::PointXYZI& point = result.keypoints->operator[](i);
                const auto& kp = keypoints[i];

                point.x = kp.pos.x;
                point.y = kp.pos.y;
                point.z = kp.pos.z;
                point.intensity = kp.score; 
            }
        }
        return result;
    }

    gpu_Submap CudaCore::flattenSubmap(const Submap& submap, const std::vector<Keyframe>& submap_keyframes){
        gpu_Submap result;
        size_t num_keyframe = submap_keyframes.size();
        auto keyframe_vector = submap_keyframes;
        
        size_t num_points = 0;
        std::vector<roller::PointCloudInt> correctedKeyframes;
        
        for(size_t i = 0; i < num_keyframe; i++){
            const auto cloud = keyframe_vector[i].pointcloud;
            if (!cloud) {
                std::cerr << "[CudaCore] : Null pointcloud detected in keyframe " << i << ". Skipping." << std::endl;
                continue; 
            }
            const auto correction = submap.correction;
            PointCloudInt keyframe;
            pcl::transformPointCloud(*cloud, keyframe, correction);
            correctedKeyframes.push_back(keyframe);
            num_points += cloud->size();
        }
        result.offset_table.reserve(num_keyframe + 1);
        result.offset_table.push_back(0);
        result.submap.reserve(num_points);
        result.sensor_pos.reserve(num_keyframe);
        
        size_t offset = 0;
        float4 tmp_point;
        float3 tmp_sensor_xyz;
        for (size_t index = 0; index < num_keyframe; index++){
            for(const auto& point : correctedKeyframes[index].points){
                tmp_point.x = point.x;
                tmp_point.y = point.y;
                tmp_point.z = point.z;
                tmp_point.w = point.intensity;
                result.submap.push_back(tmp_point);
            }

            const auto &pose = submap.correction * submap.keyframe_poses[index].fg;
            Eigen::Vector3f sensor_pos_eigen = pose.block<3, 1>(0, 3);
            tmp_sensor_xyz.x = sensor_pos_eigen.x(); tmp_sensor_xyz.y = sensor_pos_eigen.y(); tmp_sensor_xyz.z = sensor_pos_eigen.z();
            result.sensor_pos.push_back(tmp_sensor_xyz);

            offset += correctedKeyframes[index].size();
            result.offset_table.push_back(offset);
        }
        result.submap_bytes = num_points * sizeof(float4);
        result.offset_table_bytes = result.offset_table.size() * sizeof(int);
        result.sensor_pos_bytes = result.sensor_pos.size() * sizeof(float3);

        result.num_keyframe = num_keyframe;
        result.num_points = num_points; 
        return result;
    }

    inline nanovdb::Coord unpackKeyToCoord(uint64_t key) {
        int x = static_cast<int>((key >> 42) & 0x1FFFFF) - 1048576;
        int y = static_cast<int>((key >> 21) & 0x1FFFFF) - 1048576;
        int z = static_cast<int>(key & 0x1FFFFF) - 1048576;
        return nanovdb::Coord(x, y, z);
    }

    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> CudaCore::buildGrid(const PointNormal_GPU* d_points_buffer, int num_points, float voxel_size){
        thrust::device_vector<uint64_t> d_keys(num_points);
        thrust::device_vector<uint32_t> d_indices(num_points);

        uint64_t* raw_keys_ptr = thrust::raw_pointer_cast(d_keys.data());
        uint32_t* raw_indices_ptr = thrust::raw_pointer_cast(d_indices.data());
        
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            std::cerr << "[buildGrid] : Failed to create CUDA stream" << std::endl;
            return nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>();
        }
        
        int threads = 512;
        int blocks = (num_points + threads - 1) / threads;
        cuda::extractVoxelKeysKernel<<<blocks, threads, 0, stream>>>(d_points_buffer, num_points, voxel_size, raw_keys_ptr, raw_indices_ptr);
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "[buildGrid] : Failed to synchronize CUDA stream" << std::endl;
            cudaStreamDestroy(stream);
            return nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>();
        }
        cudaStreamDestroy(stream);

        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());

        auto new_end = thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());
        size_t num_unique_voxels = new_end.first - d_keys.begin();

        std::vector<uint64_t> h_keys(num_unique_voxels);
        std::vector<uint32_t> h_indices(num_unique_voxels);
        thrust::copy(d_keys.begin(), new_end.first, h_keys.begin());
        thrust::copy(d_indices.begin(), new_end.second, h_indices.begin());

        uint32_t background = 0xFFFFFFFFu;
        using GridT = nanovdb::tools::build::Grid<uint32_t>; 
        auto build_grid_ptr = std::make_unique<GridT>(background);

        auto acc = build_grid_ptr->getAccessor();

        for (size_t i = 0; i < num_unique_voxels; ++i) {
            auto coord = unpackKeyToCoord(h_keys[i]);
            acc.setValue(coord, static_cast<uint32_t>(h_indices[i]));
        }

        auto handle = nanovdb::tools::createNanoGrid<GridT, uint32_t, nanovdb::cuda::DeviceBuffer>(*build_grid_ptr);
        handle.deviceUpload(0, true);

        return handle; 
    }

    std::vector<keypoints> CudaCore::keypoint_estimation_step(const nanovdb::UInt32Grid* normalMappingGrid, const PointNormal_GPU* normalBuffer, int hostNormalCount, const std::vector<PointNormal_GPU>& host_normals, std::vector<float>& kp_scores) {
        std::vector<keypoints> keypoints;
        if (Config::estimate_keypoints::method == "harris3d_cuda") {
            if (hostNormalCount > 0) {
                float harrisRadius = static_cast<float>(Config::estimate_keypoints::harris3d::radius);
                if (harrisRadius < _VoxelSize * 3.0f) harrisRadius = _VoxelSize * 3.0f;

                auto start_score = std::chrono::high_resolution_clock::now();
                float* d_scores = cuda::compute_harris3d_scores(normalMappingGrid, normalBuffer, hostNormalCount, _VoxelSize, harrisRadius);
                std::chrono::duration<double, std::milli> duration_score = std::chrono::high_resolution_clock::now() - start_score;
                std::cout << "[CudaCore] : Harris3D CUDA scores computation finished in " << duration_score.count() << " ms" << std::endl;

                if (Config::debug::allow_pointcloud_extraction) {
                    kp_scores.resize(static_cast<size_t>(hostNormalCount));
                    cudaMemcpy(kp_scores.data(), d_scores, static_cast<size_t>(hostNormalCount) * sizeof(float), cudaMemcpyDeviceToHost);
                }

                int max_kp = Config::estimate_keypoints::nb_of_selected;
                if (Config::estimate_keypoints::filtering_method == "nms") {
                    float score_thresh = static_cast<float>(Config::estimate_keypoints::harris3d_cuda::score_threshold);
                    float nms_rad = static_cast<float>(Config::estimate_keypoints::nms::radius);
                    keypoints = cuda::filter_keypoints_nms(normalMappingGrid, normalBuffer, d_scores, nullptr, hostNormalCount, nms_rad, _VoxelSize, score_thresh, max_kp);
                } else if (Config::estimate_keypoints::filtering_method == "threshold") {
                    float score_thresh = static_cast<float>(Config::estimate_keypoints::harris3d_cuda::score_threshold);
                    keypoints = cuda::filter_keypoints_threshold(normalBuffer, d_scores, nullptr, max_kp, hostNormalCount, score_thresh);
                } else {
                    keypoints = cuda::filter_keypoints_number(normalBuffer, d_scores, nullptr, max_kp, hostNormalCount);
                }
                if (d_scores) cudaFree(d_scores);
            }
        } else if (Config::estimate_keypoints::method == "iss3d_cuda") {
            if (hostNormalCount > 0) {
                float issRadius = static_cast<float>(Config::estimate_keypoints::iss3d_cuda::searchRadius);
                if (issRadius < _VoxelSize * 3.0f) issRadius = _VoxelSize * 3.0f;
                float g21 = static_cast<float>(Config::estimate_keypoints::iss3d_cuda::gamma_21);
                float g32 = static_cast<float>(Config::estimate_keypoints::iss3d_cuda::gamma_32);

                auto start_score = std::chrono::high_resolution_clock::now();
                float* d_scores = cuda::compute_iss3d_scores(normalMappingGrid, normalBuffer, hostNormalCount, _VoxelSize, issRadius, g21, g32);
                std::chrono::duration<double, std::milli> duration_score = std::chrono::high_resolution_clock::now() - start_score;
                std::cout << "[CudaCore] : ISS3D CUDA scores computation finished in " << duration_score.count() << " ms" << std::endl;

                if (Config::debug::allow_pointcloud_extraction) {
                    kp_scores.resize(static_cast<size_t>(hostNormalCount));
                    cudaMemcpy(kp_scores.data(), d_scores, static_cast<size_t>(hostNormalCount) * sizeof(float), cudaMemcpyDeviceToHost);
                }

                int max_kp = Config::estimate_keypoints::nb_of_selected;
                keypoints = cuda::filter_keypoints_number(normalBuffer, d_scores, nullptr, max_kp, hostNormalCount);
                if (d_scores) cudaFree(d_scores);
            }
        } else if (Config::estimate_keypoints::method == "harris3d_pcl_cpu") {
            keypoints = cuda::compute_pcl_cpu_harris3d(normalBuffer, hostNormalCount, host_normals, _VoxelSize);
        } else if (Config::estimate_keypoints::method == "iss3d_pcl_cpu") {
            keypoints = cuda::compute_pcl_cpu_iss3d(normalBuffer, hostNormalCount, host_normals, _VoxelSize);
        }
        return keypoints;
    }

    CudaResult CudaCore::computeDescriptors(std::vector<keypoints>& keypoints, const PointNormal_GPU* normalBuffer, int hostNormalCount, const std::vector<PointNormal_GPU>& host_normals, const std::vector<float>& kp_scores) {
        CudaResult R = this->results(host_normals, keypoints, Config::debug::allow_pointcloud_extraction, kp_scores);
        if (keypoints.empty()) {
            std::cout << "[CudaCore] : Feature descriptors not computed : No keypoints found" << std::endl;
            return R;
        }

        if (Config::extract_feature_descriptors::method == "fpfh") {
            auto start_desc = std::chrono::high_resolution_clock::now();
            bool use_k = Config::extract_feature_descriptors::fpfh::use_k_instead_of_radius;
            int k = Config::extract_feature_descriptors::fpfh::fpfh_k;
            float fpfhRadius = Config::extract_feature_descriptors::fpfh::fpfh_radius;
            auto fpfh_cloud = cuda::computeFPFH(keypoints, normalBuffer, hostNormalCount, host_normals, k, fpfhRadius, _VoxelSize, use_k);
            R.descriptors = fpfh_cloud;
            std::chrono::duration<double, std::milli> duration_desc = std::chrono::high_resolution_clock::now() - start_desc;
            std::cout << "[CudaCore] : FPFH descriptors Estimation finished in " << duration_desc.count() << " ms" << std::endl;
            std::cout << "[CudaCore] : FPFH descriptors found = " << fpfh_cloud->size() << std::endl;
            std::cout << "[CudaCore] : New Keypoint size = " << keypoints.size() << std::endl;
        } 
        else if (Config::extract_feature_descriptors::method == "shot") {
            auto start_desc = std::chrono::high_resolution_clock::now();
            float shotRadius = 0.5f; 
            auto shot_cloud = cuda::computeSHOT(keypoints, normalBuffer, hostNormalCount, host_normals, shotRadius, _VoxelSize);
            R.descriptors = shot_cloud;
            std::chrono::duration<double, std::milli> duration_desc = std::chrono::high_resolution_clock::now() - start_desc;
            std::cout << "[CudaCore] : SHOT descriptors Estimation finished in " << duration_desc.count() << " ms" << std::endl;
            std::cout << "[CudaCore] : SHOT descriptors found = " << shot_cloud->size() << std::endl;
            std::cout << "[CudaCore] : New Keypoint size = " << keypoints.size() << std::endl;
        } 
        else if (Config::extract_feature_descriptors::method == "hpnvd") {
            auto start_desc = std::chrono::high_resolution_clock::now();
            float hpnvdRadius = Config::extract_feature_descriptors::hpnvd::hpnvd_radius;
            int binNum = Config::extract_feature_descriptors::hpnvd::hpnvd_bin_num;
            auto hpnvd_cloud = cuda::computeHPNVD(keypoints, normalBuffer, hostNormalCount, host_normals, hpnvdRadius, binNum, _VoxelSize);
            R.descriptors = hpnvd_cloud;
            std::chrono::duration<double, std::milli> duration_desc = std::chrono::high_resolution_clock::now() - start_desc;
            std::cout << "[CudaCore] : HPNVD descriptors Estimation finished in " << duration_desc.count() << " ms" << std::endl;
            std::cout << "[CudaCore] : HPNVD descriptors found = " << hpnvd_cloud->size() << std::endl;
            std::cout << "[CudaCore] : New Keypoint size = " << keypoints.size() << std::endl;
        }
        else {
            std::cerr << "[CudaCore] : Unknown descriptor method specified in config." << std::endl;
        }
        return R;
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr CudaCore::importSurfaceCloud(const PointNormal_GPU* normalBuffer, int pointCount, const std::vector<PointNormal_GPU>& host_normals, float voxel_size) {
        if (cached_surface_cloud_ && !cached_surface_cloud_->empty()) return cached_surface_cloud_;
        cached_surface_cloud_ = cuda::importSurfaceCloud(normalBuffer, pointCount, host_normals, voxel_size);
        return cached_surface_cloud_;
    }

} // namespace roller::loopclosure