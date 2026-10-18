#pragma once

#include <roller/tsdf/tsdf_data.hpp>
#include <roller/submap.hpp>
#include <roller/loopclosure/pcl_toolbox.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/version.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/filter.h>

#include <chrono>
#include <sstream>

#include <Eigen/Geometry> 

#include <boost/signals2.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

class Base_node;

namespace roller::loopclosure{

    using RoPSSignature = pcl::Histogram<135>;

    class loop_closure{
        public:
            loop_closure(Base_node *node);
            void readconfig();

            //Signal handling
            void on_start_pipeline();

        private:
            Base_node* _node;
            
            int _number_of_iterations, _extraction_method, _ransac_max_iter;
            double _voxel_size, _extraction_strength, _fpfh_radius, _ransac_inlier_threshold, _noise_intensity;
            bool _files_ok = true;
            bool _is_a_test, _reciprocal_match, _use_external_as_input, _use_external_as_reference, _add_noise;
            std::string _path_to_input, _path_to_reference;
            
            pcl::PointCloud<pcl::PointNormal>::Ptr _src_input_cloud, _src_reference_cloud;
            pcl::PointCloud<pcl::PointNormal>::Ptr _input_cloud, _reference_cloud, _transformed_cloud;
            pcl::PointCloud<pcl::PointXYZI>::Ptr _input_keypoints, _reference_keypoints;
            pcl::CorrespondencesPtr _correspondences, _filtered_correspondences;

            Eigen::Vector4f _input_centroid, _reference_centroid;
            Eigen::Matrix4f _modif_transform = Eigen::Matrix4f::Identity();

            pcl::search::KdTree<pcl::PointXYZI>::Ptr _tree;

            Eigen::Matrix4f _TransformationMatrix;
            void get_pointcloud();
            void move_reference(float x_angle, float y_angle, float z_angle, float x_translation, float y_translation, float z_translation);
            void center_pointclouds();
            
            pcl::PointCloud<pcl::PointXYZI>::Ptr estimate_keypoints(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, int method, float strength);
            pcl::PointCloud<pcl::FPFHSignature33>::Ptr extract_fpfh_descriptors(pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints, pcl::PointCloud<pcl::PointNormal>::Ptr full_cloud);
            pcl::PointCloud<RoPSSignature>::Ptr extract_rops_descriptors(pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints, pcl::PointCloud<pcl::PointNormal>::Ptr full_cloud);
            
            void match_features(pcl::PointCloud<pcl::FPFHSignature33>::Ptr input, pcl::PointCloud<pcl::FPFHSignature33>::Ptr reference);

            void match_features(pcl::PointCloud<RoPSSignature>::Ptr input, pcl::PointCloud<RoPSSignature>::Ptr reference);

            void match_rejection(pcl::CorrespondencesPtr correspondences);
            Eigen::Matrix4f estimate_transform(pcl::CorrespondencesPtr correspondences);


            void reset_object();
            void loop_cleaning();
            void publish_debug_pipeline();

    };
}