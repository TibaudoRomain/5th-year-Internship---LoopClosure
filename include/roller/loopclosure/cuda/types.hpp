#pragma once

#include <array>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <roller/loopclosure/Config.hpp>

namespace roller::loopclosure {

    /**
     * @brief Structure representing a 3D surface point with normal and curvature on GPU.
     */
    struct PointNormal_GPU {
        float3 pos;           ///< 3D position (x, y, z) of the surface point in world/submap coordinates.
        float3 normal;        ///< Unit surface normal vector (nx, ny, nz) estimated from TSDF gradient.
        float normal_trust;   ///< Confidence measure of the normal estimation quality (0.0 to 1.0).
        float curvature;      ///< Estimated local surface curvature score.
    };
    /**
     * @brief Structure representing a 3D surface point with normal and curvature on GPU.
     */
    struct Point_GPU {
        float3 pos;           ///< 3D position (x, y, z) of the surface point in world/submap coordinates.
    };

    /**
     * @brief Structure representing an extracted 3D keypoint with response score and scale.
     */
    struct keypoints {
        float3 pos;           ///< 3D position (x, y, z) of the keypoint.
        float3 normal;        ///< Surface normal vector (nx, ny, nz) at keypoint location.
        float normal_trust;   ///< Confidence score of the normal vector.
        float curvature;      ///< Local surface curvature at keypoint location.
        
        float score;          ///< Keypoint detector response score (e.g. Harris3D or ISS3D response).
        float scale;          ///< Detection scale / radius associated with the keypoint.
        int index_on_map;     ///< Index of the keypoint within the original surface point cloud array.
    };

    /**
     * @brief Container structure for flattening submaps and keyframes prior to GPU upload.
     */
    struct gpu_Submap {
        std::vector<float4> submap;         ///< Flattened keyframe point cloud array (x, y, z, intensity).
        size_t submap_bytes;               ///< Memory size of submap points in bytes.
        
        std::vector<int> offset_table;       ///< Keyframe point offset boundaries within the submap vector.
        size_t offset_table_bytes;         ///< Memory size of offset table in bytes.

        std::vector<float3> sensor_pos;      ///< Sensor position (x, y, z) for each keyframe origin.
        size_t sensor_pos_bytes;           ///< Memory size of sensor positions in bytes.

        size_t num_keyframe;               ///< Total number of keyframes in the submap.
        size_t num_points;                 ///< Total number of points in the submap.
    };

    /**
     * @brief Result container produced by the device loopclosure processing pipeline.
     */
    struct CudaResult {
        pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints;          ///< Extracted keypoints cloud with intensity storing response scores.
        pcl::PointCloud<pcl::PointNormal>::Ptr normals;           ///< Surface point cloud containing estimated normals and curvatures.
        pcl::PointCloud<pcl::PointXYZI>::Ptr debug_harrisscore;   ///< Optional debug point cloud containing all keypoint detector scores.

        Config::DescriptorCloudPtr descriptors;                  ///< Extracted 3D descriptors (e.g. FPFH / HPNVD / SHOT variant).

        ~CudaResult() = default;
    };

    namespace cuda {

        /**
         * @brief Dynamic Patch structure representing a local 3D neighborhood around a query point.
         * Surface point indices are stored in a contiguous device buffer starting at index `offset`.
         */
        struct Patch {
            float3 query_pt; ///< 3D position (x, y, z) of the query center point.
            int count;       ///< Number of surface points contained inside the patch radius.
            int offset;      ///< Starting index into the global contiguous patch index array.
            float score;
        };

        /**
         * @brief 3x3 Symmetric Covariance Matrix representation for a local point patch.
         * Stores spatial centroid and upper-triangular covariance elements (cxx, cxy, cxz, cyy, cyz, czz).
         */
        struct PatchCovariance {
            float3 centroid;  ///< Mean position (x, y, z) of points inside the patch.
            float cxx;        ///< Covariance element (0,0).
            float cxy;        ///< Covariance element (0,1).
            float cxz;        ///< Covariance element (0,2).
            float cyy;        ///< Covariance element (1,1).
            float cyz;        ///< Covariance element (1,2).
            float czz;        ///< Covariance element (2,2).
            int point_count;  ///< Number of points used to compute covariance.
        };

        /**
         * @brief 16-byte aligned leaf node structure for zero-copy GPU vector transfer (ld.global.v4.f32).
         * Optimized to exactly 16 bytes (4x float32), removing CPU-only fields (max_score, point_count).
         */
        struct alignas(16) LeafNodeInfo {
            float center_x;   ///< Leaf center X coordinate in meters (4 bytes)
            float center_y;   ///< Leaf center Y coordinate in meters (4 bytes)
            float center_z;   ///< Leaf center Z coordinate in meters (4 bytes)
            float size;       ///< Leaf cube side length in meters (4 bytes)

            __host__ __device__ inline float3 get_center() const {
                return make_float3(center_x, center_y, center_z);
            }

            __host__ __device__ inline float get_radius() const {
                constexpr float HALF_SQRT_3 = 0.8660254037844386f;
                return size * HALF_SQRT_3;
            }
        };

        /**
         * @brief Structure representing a filled Octree Leaf containing attributed surface point indices on GPU.
         * Surface point indices are stored in a contiguous device buffer starting at index `offset`.
         */
        struct FilledLeaf {
            float3 center;   ///< 3D position (x, y, z) of the leaf center point.
            float size;      ///< Leaf cube side length in meters.
            int count;       ///< Number of surface points contained inside the leaf box.
            int offset;      ///< Starting index into the global contiguous leaf point index array.
        };

    } // namespace cuda

} // namespace roller::loopclosure
