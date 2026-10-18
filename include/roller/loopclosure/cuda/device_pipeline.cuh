#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <roller/submap.hpp>
#include <roller/loopclosure/cuda/types.hpp>
#include <roller/loopclosure/Config.hpp>

#undef NANOVDB_USE_OPENVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#include <roller/loopclosure/cuda/descriptors.cuh>
#include <roller/loopclosure/cuda/keypoints.cuh>
#include <roller/loopclosure/cuda/normals.cuh>
#include <roller/loopclosure/cuda/leaf.cuh>
#include <roller/loopclosure/cuda/pointcloud.cuh>

namespace roller::loopclosure {

    const int AXIS_X = 0;
    const int AXIS_Y = 1;
    const int AXIS_Z = 2;

    class CudaCore {
        public:
            CudaCore(float voxel_size);
            ~CudaCore();
            
            CudaResult LDPipeline_device(const Submap& submap, const std::vector<Keyframe>& submap_keyframes, const nanovdb::GridHandle<nanovdb::HostBuffer>& sdf);
        
        private:       
            float _VoxelSize;
        
            float4* gpu_Submap_Ptr;
            float3* gpu_SensorPos_Ptr;
            int* gpu_SubmapOffsets_Ptr;

            void* gpu_SDFHandle_Ptr;
            nanovdb::FloatGrid* gpu_SDF_Ptr;
            
            PointNormal_GPU* normalbuffer;
            int* normalCounter;
            pcl::PointCloud<pcl::PointNormal>::Ptr cached_surface_cloud_ = nullptr;

            nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> buildGrid(const PointNormal_GPU* d_points_buffer, int num_points, float voxel_size);
            gpu_Submap flattenSubmap(const Submap& submap, const std::vector<Keyframe>& submap_keyframes);
            
            CudaResult empty_result();
            CudaResult results(const std::vector<PointNormal_GPU>& host_normals, const std::vector<keypoints>& keypoints, bool allow_pointcloud_extraction, const std::vector<float>& kp_scores);
            std::vector<keypoints> keypoint_estimation_step(const nanovdb::UInt32Grid* normalMappingGrid, const PointNormal_GPU* normalBuffer, int hostNormalCount, const std::vector<PointNormal_GPU>& host_normals, std::vector<float>& kp_scores);
            CudaResult computeDescriptors(std::vector<keypoints>& keypoints, const PointNormal_GPU* normalBuffer, int hostNormalCount, const std::vector<PointNormal_GPU>& host_normals, const std::vector<float>& kp_scores);
            
            pcl::PointCloud<pcl::PointNormal>::Ptr importSurfaceCloud(const PointNormal_GPU* normalBuffer, int pointCount, const std::vector<PointNormal_GPU>& host_normals, float voxel_size);
    };

} // namespace roller::loopclosure