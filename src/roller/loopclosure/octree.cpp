#include "roller/loopclosure/octree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

namespace roller::loopclosure {

    namespace {
        void collectLeafs(const OctreeNode* node, std::vector<LeafNodeInfo>& leafs) {
            if (!node) return;
            if (node->is_leaf) {
                leafs.push_back({node->center_x, node->center_y, node->center_z, node->size});
                return;
            }

            for (const auto& child : node->children) {
                if (child) {
                    collectLeafs(child.get(), leafs);
                }
            }
        }
    }

    std::vector<LeafNodeInfo> CustomOctree::extractLeafs(const std::vector<std::unique_ptr<OctreeNode>>& roots) {
        std::vector<LeafNodeInfo> leafs;
        size_t estimated_capacity = 0;
        for (const auto& root : roots) {
            if (root) estimated_capacity += root->is_leaf ? 1 : 8;
        }
        leafs.reserve(estimated_capacity);

        for (const auto& root : roots) {
            collectLeafs(root.get(), leafs);
        }
        return leafs;
    }

    void CustomOctree::extractLeafCenters(
        const std::vector<std::unique_ptr<OctreeNode>>& roots,
        std::vector<float3>& leaf_centers,
        std::vector<float>& leaf_radii
    ) {
        std::vector<LeafNodeInfo> leafs = extractLeafs(roots);
        leaf_centers.clear();
        leaf_radii.clear();
        leaf_centers.reserve(leafs.size());
        leaf_radii.reserve(leafs.size());

        for (const auto& leaf : leafs) {
            leaf_centers.push_back(leaf.get_center());
            leaf_radii.push_back(leaf.get_radius());
        }
    }

    std::vector<std::unique_ptr<OctreeNode>> CustomOctree::build(const pcl::PointCloud<pcl::PointXYZI>& pcl_cloud) {
        PointCloud4D cloud;
        cloud.reserve(pcl_cloud.size());
        for (const auto& pt : pcl_cloud) {
            cloud.push_back(pt.x, pt.y, pt.z, pt.intensity);
        }
        return build(cloud);
    }

    std::vector<std::unique_ptr<OctreeNode>> CustomOctree::build(const PointCloud4D& cloud) {
        std::vector<std::unique_ptr<OctreeNode>> roots;
        if (cloud.empty()) return roots;

        // Compute 3D bounding box
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        float max_z = std::numeric_limits<float>::lowest();

        size_t n = cloud.size();
        for (size_t i = 0; i < n; ++i) {
            min_x = std::min(min_x, cloud.x[i]);
            min_y = std::min(min_y, cloud.y[i]);
            min_z = std::min(min_z, cloud.z[i]);
            max_x = std::max(max_x, cloud.x[i]);
            max_y = std::max(max_y, cloud.y[i]);
            max_z = std::max(max_z, cloud.z[i]);
        }

        // Align root grid to root_size boxes (configurable)
        const float root_size = std::max(config_.root_size, 0.1f);
        int grid_min_x = static_cast<int>(std::floor(min_x / root_size));
        int grid_max_x = static_cast<int>(std::floor(max_x / root_size));
        int grid_min_y = static_cast<int>(std::floor(min_y / root_size));
        int grid_max_y = static_cast<int>(std::floor(max_y / root_size));
        int grid_min_z = static_cast<int>(std::floor(min_z / root_size));
        int grid_max_z = static_cast<int>(std::floor(max_z / root_size));

        int num_x = grid_max_x - grid_min_x + 1;
        int num_y = grid_max_y - grid_min_y + 1;
        int num_z = grid_max_z - grid_min_z + 1;

        std::vector<std::unique_ptr<OctreeNode>> grid(num_x * num_y * num_z);

        for (int gx = 0; gx < num_x; ++gx) {
            for (int gy = 0; gy < num_y; ++gy) {
                for (int gz = 0; gz < num_z; ++gz) {
                    float cx = (grid_min_x + gx + 0.5f) * root_size;
                    float cy = (grid_min_y + gy + 0.5f) * root_size;
                    float cz = (grid_min_z + gz + 0.5f) * root_size;
                    int idx = gx + gy * num_x + gz * num_x * num_y;
                    grid[idx] = std::make_unique<OctreeNode>(cx, cy, cz, root_size);
                }
            }
        }

        // Assign points to root boxes
        for (int i = 0; i < static_cast<int>(n); ++i) {
            int gx = static_cast<int>(std::floor(cloud.x[i] / root_size)) - grid_min_x;
            int gy = static_cast<int>(std::floor(cloud.y[i] / root_size)) - grid_min_y;
            int gz = static_cast<int>(std::floor(cloud.z[i] / root_size)) - grid_min_z;

            if (gx >= 0 && gx < num_x && gy >= 0 && gy < num_y && gz >= 0 && gz < num_z) {
                int idx = gx + gy * num_x + gz * num_x * num_y;
                grid[idx]->point_indices.push_back(i);
            }
        }

        // Subdivide non-empty root boxes starting at division level 0
        for (auto& root : grid) {
            if (!root->point_indices.empty()) {
                subdivideNode(root.get(), cloud, 0);
                roots.push_back(std::move(root));
            }
        }

        return roots;
    }

    void CustomOctree::subdivideNode(OctreeNode* node, const PointCloud4D& cloud, int current_division) {
        if (!node || node->point_indices.empty()) return;

        // Calculate maximum score in node area
        float max_score = 0.0f;
        for (int idx : node->point_indices) {
            max_score = std::max(max_score, cloud.score[idx]);
        }
        node->max_score = max_score;
        node->point_count = static_cast<int>(node->point_indices.size());

        // Check maximum number of divisions (depth limit)
        if (current_division >= config_.max_divisions) {
            node->is_leaf = true;
            return;
        }

        // Rule 1: Check minimum cube side length
        if (node->size <= config_.min_cube_size + 1e-4f) {
            node->is_leaf = true;
            return;
        }

        // Rule 2: Check minimum number of points in node area
        if (node->point_count < config_.min_point_count) {
            node->is_leaf = true;
            return;
        }

        // Configurable score division thresholds matching division level
        float req_threshold = 0.0f;
        if (current_division == 0) {
            req_threshold = config_.score_threshold_4m_to_2m;
        } else if (current_division == 1) {
            req_threshold = config_.score_threshold_2m_to_1m;
        } else if (current_division == 2) {
            req_threshold = config_.score_threshold_1m_to_0_5m;
        } else {
            req_threshold = config_.score_threshold_1m_to_0_5m;
        }

        if (node->max_score < req_threshold) {
            node->is_leaf = true;
            return;
        }

        // Subdivide into 8 octants
        float child_size = node->size * 0.5f;
        float offset = node->size * 0.25f;

        for (int octant = 0; octant < 8; ++octant) {
            float child_cx = node->center_x + ((octant & 1) ? offset : -offset);
            float child_cy = node->center_y + ((octant & 2) ? offset : -offset);
            float child_cz = node->center_z + ((octant & 4) ? offset : -offset);

            node->children[octant] = std::make_unique<OctreeNode>(child_cx, child_cy, child_cz, child_size);
        }

        // Assign point indices to child octants
        for (int idx : node->point_indices) {
            int octant = 0;
            if (cloud.x[idx] >= node->center_x) octant |= 1;
            if (cloud.y[idx] >= node->center_y) octant |= 2;
            if (cloud.z[idx] >= node->center_z) octant |= 4;

            node->children[octant]->point_indices.push_back(idx);
        }

        node->is_leaf = false;

        // Recursively subdivide child octants with incremented division level
        for (int octant = 0; octant < 8; ++octant) {
            if (!node->children[octant]->point_indices.empty()) {
                subdivideNode(node->children[octant].get(), cloud, current_division + 1);
            }
        }
    }

} // namespace roller::loopclosure
