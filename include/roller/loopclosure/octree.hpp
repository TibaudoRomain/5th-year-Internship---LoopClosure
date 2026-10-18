#pragma once

#include <array>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <roller/loopclosure/cuda/types.hpp>

namespace roller::loopclosure {

    using cuda::LeafNodeInfo;

    /**
     * @brief High-Performance Structure of Arrays (SoA) 4-channel point cloud container (x, y, z, score).
     * Enables 8x-16x SIMD vectorization (AVX-256 / AVX-512), 100% L1 cache line utilization, and zero-copy GPU DMA.
     */
    struct PointCloud4D {
        std::vector<float> x;      ///< Contiguous X positions in meters
        std::vector<float> y;      ///< Contiguous Y positions in meters
        std::vector<float> z;      ///< Contiguous Z positions in meters
        std::vector<float> score;  ///< Contiguous feature / salience scores

        size_t size() const { return x.size(); }
        bool empty() const { return x.empty(); }
        void clear() { x.clear(); y.clear(); z.clear(); score.clear(); }
        void reserve(size_t n) { x.reserve(n); y.reserve(n); z.reserve(n); score.reserve(n); }
        void push_back(float px, float py, float pz, float ps) {
            x.push_back(px);
            y.push_back(py);
            z.push_back(pz);
            score.push_back(ps);
        }
    };

    /**
     * @brief Configuration parameters for CustomOctree division rules.
     */
    struct OctreeConfig {
        float root_size = 4.0f;                  ///< Initial root box side length in meters (default: 4.0m)
        int min_point_count = 10;                ///< Minimum number of points in a node to allow division (default: 10)
        int max_divisions = 3;                   ///< Maximum division levels allowed (depth limit, default: 3)
        float min_cube_size = 0.5f;              ///< Minimum cube side length in meters (default: 0.5m)

        float score_threshold_4m_to_2m = 0.12f;  ///< Threshold to divide root level (4m -> 8x 2m)
        float score_threshold_2m_to_1m = 0.36f;  ///< Threshold to divide level 1 (2m -> 8x 1m)
        float score_threshold_1m_to_0_5m = 0.60f;///< Threshold to divide level 2 (1m -> 8x 0.5m)
    };

    /**
     * @brief Octree Node structure representing a 3D cubic region.
     */
    struct OctreeNode {
        float center_x;  ///< Center X coordinate of node cube in meters
        float center_y;  ///< Center Y coordinate of node cube in meters
        float center_z;  ///< Center Z coordinate of node cube in meters
        float size;      ///< Cube side length in meters (4.0, 2.0, 1.0, 0.5)

        float max_score;                ///< Maximum score among all points inside this node area
        int point_count;                ///< Total number of points inside this node area
        std::vector<int> point_indices; ///< Indices of points assigned to this node area

        bool is_leaf;                                          ///< True if node is not subdivided
        std::array<std::unique_ptr<OctreeNode>, 8> children;    ///< 8 child octants (nullptr if leaf)

        OctreeNode(float cx, float cy, float cz, float sz)
            : center_x(cx), center_y(cy), center_z(cz), size(sz),
              max_score(0.0f), point_count(0), is_leaf(true) {
            for (auto& child : children) child = nullptr;
        }
    };

    /**
     * @brief Custom CPU Octree builder enforcing hierarchical score & density division rules.
     */
    class CustomOctree {
    public:
        CustomOctree() = default;
        explicit CustomOctree(const OctreeConfig& config) : config_(config) {}
        ~CustomOctree() = default;

        /**
         * @brief Get runtime configuration options.
         */
        const OctreeConfig& getConfig() const { return config_; }

        /**
         * @brief Set runtime configuration options.
         */
        void setConfig(const OctreeConfig& config) { config_ = config; }

        /**
         * @brief Builds custom octree from high-performance SoA PointCloud4D (x, y, z, score).
         * 
         * @param cloud Input SoA point cloud container
         * @return Vector of root node pointers covering the dataset bounding box
         */
        std::vector<std::unique_ptr<OctreeNode>> build(const PointCloud4D& cloud);

        /**
         * @brief Overload for PCL PointXYZI point clouds (intensity channel stores score).
         */
        std::vector<std::unique_ptr<OctreeNode>> build(const pcl::PointCloud<pcl::PointXYZI>& cloud);

        /**
         * @brief Recursively traverses octree roots and extracts all leaf nodes.
         * 
         * @param roots Vector of root OctreeNode pointers
         * @return Vector of 16-byte aligned LeafNodeInfo for all leaf boxes
         */
        static std::vector<LeafNodeInfo> extractLeafs(const std::vector<std::unique_ptr<OctreeNode>>& roots);

        /**
         * @brief Extracts leaf center positions (float3) and search radii for GPU Spatial Hashing.
         * 
         * @param roots Vector of root OctreeNode pointers
         * @param leaf_centers Output vector of float3 centers (x, y, z) for GPU query buffer
         * @param leaf_radii Output vector of query search radii for each leaf (circumscribed sphere radius: size * sqrt(3) / 2)
         */
        static void extractLeafCenters(
            const std::vector<std::unique_ptr<OctreeNode>>& roots,
            std::vector<float3>& leaf_centers,
            std::vector<float>& leaf_radii
        );

    private:
        void subdivideNode(OctreeNode* node, const PointCloud4D& cloud, int current_division);

        OctreeConfig config_{};
    };

} // namespace roller::loopclosure
