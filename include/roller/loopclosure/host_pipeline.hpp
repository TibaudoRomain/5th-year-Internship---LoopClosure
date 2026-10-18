#pragma once

#include <roller/tsdf/tsdf_data.hpp>
#include <roller/submap.hpp>
#include <roller/loopclosure/Config.hpp>
#include <roller/loopclosure/pcl_toolbox.hpp>
#include <roller/loopclosure/cuda/device_pipeline.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>


#include <rclcpp/time.hpp>
#include <rclcpp/publisher.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>


#include <chrono>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <teaser/registration.h>

#include <boost/signals2.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>




#include <roller/loopclosure/descriptors/ndtmc.hpp>
#include <roller/loopclosure/benchmark.hpp>

class Base_node;

namespace roller::loopclosure{

    struct references{
        size_t submap_db_id;
        Eigen::Matrix4f fg;
        Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
        Eigen::Vector3f pose;
        rclcpp::Time t_stamp = rclcpp::Time(0); //time stamp of the submap

        pcl::PointCloud<pcl::PointNormal>::Ptr normals;
        pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints;

        Config::DescriptorCloudPtr descriptors;

        pcl::PointCloud<pcl::PointXYZI>::Ptr debug_harrisscore;

        Config::DynamicDescriptor global_descriptor;

        Eigen::MatrixXd ndtmc_descriptor;
    };

    struct transformations{
        references submap_a;
        references submap_b;
        Eigen::Matrix4f transformation;
        int nb_correspondences;
    };
    
    
    using RoPSSignature = pcl::Histogram<135>;

    class loopclosure_master{
        public:
            loopclosure_master(rclcpp::Node *node);
            void readconfig();

            void set_saving_directory(const std::string& saving_dir);

            void download_submaps_description();
            
            void LDPipeline_host(const Submap& submap, size_t submap_db_id, 
                const std::vector<Keyframe>& submap_keyframes, 
                const nanovdb::GridHandle<nanovdb::HostBuffer>& sdf);

            const LoopClosureBenchmark& get_benchmark() const { return benchmark; }

        private:
            rclcpp::Node *_node;
            LoopClosureBenchmark benchmark;
            
            std::vector<references> submap_storage;

            bool debug_mode = false;

            // Transformation estimation parameters
            std::string _transformation_method;
            bool _teaser_estimate_scaling;
            double _teaser_noise_bound_multiplier;
            std::string _teaser_rotation_algorithm;
            double _teaser_rotation_gnc_factor;
            int _teaser_rotation_max_iterations;
            double _teaser_rotation_cost_threshold;
            std::string _teaser_inlier_selection_mode;
            std::string _teaser_inlier_graph_formulation;
            double _teaser_max_clique_time_limit;

            
            Eigen::Matrix4f _error_matrix = Eigen::Matrix4f::Zero();
            Eigen::Vector4f _input_centroid, _reference_centroid;
            Eigen::Matrix4f _modif_transform = Eigen::Matrix4f::Identity();

            pcl::search::KdTree<pcl::PointXYZI>::Ptr _tree;
            
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _new_submap_keypoints_pub;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _old_submap_keypoints_pub;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _new_submap_pub;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _old_submap_pub;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _transformed_cloud_pub;
            rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _correspondences_pub;

            std::shared_ptr<roller::loopclosure::CudaCore_Wrapper> cudacore;

            //Exclusion parameters :
                //filter preLoop : geometry check
            float _distance_margin;
            int _ImmediatePastExclusion;
                //filter postLoop : keypoints prediction check
            float _kpInliersThreshold; //in %

            teaser::RobustRegistrationSolver::Params tpp_params;

            Eigen::Matrix4f _TestMatrix, _TransformationMatrix;
            
            template<typename DescriptorT>
            void init_globalDescriptor(references& reference);

            void config_tpp_params();
            teaser::RobustRegistrationSolver::Params get_tpp_params();
            
            
            void compare_submaps_umeyama(const references& new_submap, const references& old_submap, CandidatePairEvaluation* eval = nullptr);
            void compare_submaps_teaser(const references& new_submap, const references& old_submap, CandidatePairEvaluation* eval = nullptr);

            void candidate_search(const std::vector<references>& allsubmaps, const references& new_submap, std::vector<references>& candidates, float margin);
            
            template <typename DescriptorT>
            void match_features_lr( typename pcl::PointCloud<DescriptorT>::Ptr input, 
                                    typename pcl::PointCloud<DescriptorT>::Ptr reference,
                                    pcl::Correspondences& output, double lowe_ratio);
            
            template <typename DescriptorT>
            void match_features_pcl(typename pcl::PointCloud<DescriptorT>::Ptr input, 
                                    typename pcl::PointCloud<DescriptorT>::Ptr reference, 
                                    pcl::Correspondences& output);

            template <typename DescriptorT>
            void match_features_sinkhorn(
                typename pcl::PointCloud<DescriptorT>::Ptr input, 
                typename pcl::PointCloud<DescriptorT>::Ptr reference, 
                pcl::Correspondences& output,
                float epsilon = 0.05f,       
                float dustbin_score = -0.5f, 
                int max_iterations = 20);


            void match_rejection(pcl::CorrespondencesPtr correspondences, pcl::Correspondences& output, pcl::PointCloud<pcl::PointNormal>::Ptr inputSource,  pcl::PointCloud<pcl::PointNormal>::Ptr inputTarget);
            
            Eigen::Matrix4f estimate_transform_umeyama(pcl::CorrespondencesPtr correspondences, pcl::PointCloud<pcl::PointNormal>::Ptr inputSource,  pcl::PointCloud<pcl::PointNormal>::Ptr inputTarget);
            Eigen::Matrix4f estimate_transform_teaser(pcl::CorrespondencesPtr correspondences, pcl::Correspondences& filtered_correspondences, pcl::PointCloud<pcl::PointXYZI>::Ptr sourceKeypoints, pcl::PointCloud<pcl::PointXYZI>::Ptr targetKeypoints, double voxel_size);

            void validate_candidates();

            
            void debug(const references& new_submap, const references& old_submap, const Eigen::Matrix4f& transformation);
            void publish_debug_cloud(const pcl::PointCloud<pcl::PointNormal>::ConstPtr& cloud,
                                     const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher,
                                     const std::string& frame_id,
                                     const rclcpp::Time& stamp);
            void publish_correspondences(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source_keypoints,
                                        const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& target_keypoints,
                                        const pcl::CorrespondencesPtr& all_correspondences,
                                        const pcl::Correspondences& filtered_correspondences,
                                        const std::string& frame_id,
                                        const rclcpp::Time& stamp);

            
            void reset_object();
    };
}

#include <roller/detail/host_pipeline_impl.hpp>