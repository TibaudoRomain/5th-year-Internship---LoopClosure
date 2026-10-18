#pragma once

#include <string>
#include <variant>
#include <rclcpp/rclcpp.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/shot_omp.h>
#include <roller/loopclosure/descriptors/HPNVD.hpp>

namespace roller::loopclosure::Config {
    inline double voxel_size = 0.05;
    inline double sonar_range = 15.0;

    using DynamicDescriptor = std::variant<pcl::FPFHSignature33, pcl::SHOT352, HPVNDSignature324>;
    using DescriptorCloudPtr = std::variant<pcl::PointCloud<pcl::FPFHSignature33>::Ptr, pcl::PointCloud<pcl::SHOT352>::Ptr, pcl::PointCloud<HPVNDSignature324>::Ptr>;
    
    namespace normal_calculation {
        inline bool gaussian_filter = false;
        inline int gaussian_width = 0;
        inline int gaussian_repetitions = 4;
        inline double distance_to_tsdf = 0.0f;
    }
    
    namespace estimate_keypoints {
        inline std::string method = "harris3d";
        inline std::string filtering_method = "thresholding";
        inline double keypoint_sampler_size = 4.0;
        inline bool fixed_nb_of_kp = false;
        inline int nb_of_selected = 512000;
        namespace scale_aware {
            inline bool enabled = true;
            inline int num_scales = 3;
            inline double scale_multiplier = 1.5;
        }
        namespace nms {
            inline double radius = 0.5;
        }

        namespace harris3d {
            inline double radius = 0.2;
            inline double score_threshold = 0.0;
        }

        namespace iss {
            inline double searchRadius = 0.2;
            inline double gamma_21 = 0.8;
            inline double gamma_32 = 0.5;
            inline int min_neighbors = 5;
        }

        namespace harris3d_cuda {
            inline double score_threshold = -0.0325;
            inline double k = 0.04;
        }

        namespace iss3d_cuda {
            inline double searchRadius = 0.2;
            inline double gamma_21 = 0.8;
            inline double gamma_32 = 0.5;
            inline double score_threshold = 0.002;
        }

        namespace voxelisation {
            inline double voxel_size = 0.1;
        }
    }

    namespace extract_feature_descriptors {
        inline std::string method = "fpfh";

        namespace fpfh {
            inline bool use_k_instead_of_radius = true;
            inline double fpfh_radius = 0.2;
            inline int fpfh_k = 20;
        }
        namespace shot {
            inline double shot_radius = 0.2;
        }
        namespace hpnvd {
            inline double hpnvd_radius = 0.5;
            inline int hpnvd_bin_num = 36;
        }
    }

    namespace match_features {
        inline std::string method = "base_pcl";
        inline double lowe_ratio = 0.0;
        inline bool reciprocal = false;

        namespace sinkhorn {
            inline double epsilon = 0.05;
            inline double dustbin_score = -0.5;
            inline int max_iterations = 20;
        }
    }

    namespace match_rejection {
        inline double inlier_threshold = 0.2;
        inline int max_iteration = 5000;
    }

    namespace transformation_estimation {
        inline std::string method = "umeyama";

        namespace teaser {
            inline bool estimate_scaling = false;
            inline double noise_bound_multiplier = 2.0;
            inline std::string rotation_estimation_algorithm = "GNC_TLS";
            inline double rotation_gnc_factor = 1.4;
            inline int rotation_max_iterations = 100;
            inline double rotation_cost_threshold = 1.0e-6;
            inline std::string inlier_selection_mode = "PMC_EXACT";
            inline std::string inlier_graph_formulation = "CHAIN";
            inline double max_clique_time_limit = 0.5;
        }
    }

    namespace candidate_search {
        inline std::string method = "ndtmc";
        inline double max_distance = 15.0;
        inline int temporal_exclusion = 2;
        inline double ndtmc_dist_threshold = 0.5;
        inline double ndtmc_resolution = 1.0;
    }

    namespace debug {
        // Note: Enabling allow_pointcloud_extraction will copy full surface pointclouds from GPU and impact pipeline performance.
        inline bool allow_pointcloud_extraction = false;
    }

    inline void readconfig(rclcpp::Node* node) {
        if (!node) return;

        // Root parameters
        voxel_size = node->get_parameter("voxel_size").as_double();
        sonar_range = node->get_parameter("sonar.range").as_double();

        // Normals calculations parameters
        normal_calculation::gaussian_filter = node->get_parameter("normal_calculation.gaussian_filter").as_bool();
        normal_calculation::gaussian_width = node->get_parameter("normal_calculation.gaussian_width").as_int();
        normal_calculation::gaussian_repetitions = node->get_parameter("normal_calculation.gaussian_repetitions").as_int();
        normal_calculation::distance_to_tsdf = node->get_parameter("normal_calculation.distance_to_tsdf").as_double();

        // Keypoints parameters
        estimate_keypoints::method = node->get_parameter("estimate_keypoints.method").as_string();
        estimate_keypoints::filtering_method = node->get_parameter("estimate_keypoints.filtering_method").as_string();
        estimate_keypoints::keypoint_sampler_size = node->get_parameter("estimate_keypoints.keypoint_sampler_size").as_double();
        estimate_keypoints::scale_aware::enabled = node->get_parameter("estimate_keypoints.scale_aware.enabled").as_bool();
        estimate_keypoints::scale_aware::num_scales = node->get_parameter("estimate_keypoints.scale_aware.num_scales").as_int();
        estimate_keypoints::scale_aware::scale_multiplier = node->get_parameter("estimate_keypoints.scale_aware.scale_multiplier").as_double();
        estimate_keypoints::nms::radius = node->get_parameter("estimate_keypoints.nms.radius").as_double();
        estimate_keypoints::fixed_nb_of_kp = node->get_parameter("estimate_keypoints.fixed_nb_of_kp").as_bool();
        estimate_keypoints::nb_of_selected = node->get_parameter("estimate_keypoints.nb_of_selected").as_int();

        estimate_keypoints::harris3d::radius = node->get_parameter("estimate_keypoints.harris3d.radius").as_double();
        estimate_keypoints::harris3d::score_threshold = node->get_parameter("estimate_keypoints.harris3d.score_threshold").as_double();

        estimate_keypoints::iss::searchRadius = node->get_parameter("estimate_keypoints.iss.searchRadius").as_double();
        estimate_keypoints::iss::gamma_21 = node->get_parameter("estimate_keypoints.iss.gamma_21").as_double();
        estimate_keypoints::iss::gamma_32 = node->get_parameter("estimate_keypoints.iss.gamma_32").as_double();
        estimate_keypoints::iss::min_neighbors = node->get_parameter("estimate_keypoints.iss.min_neighbors").as_int();

        estimate_keypoints::harris3d_cuda::score_threshold = node->get_parameter("estimate_keypoints.harris3d_cuda.score_threshold").as_double();
        estimate_keypoints::harris3d_cuda::k = node->get_parameter("estimate_keypoints.harris3d_cuda.k").as_double();

        estimate_keypoints::iss3d_cuda::searchRadius = node->get_parameter("estimate_keypoints.iss3d_cuda.searchRadius").as_double();
        estimate_keypoints::iss3d_cuda::gamma_21 = node->get_parameter("estimate_keypoints.iss3d_cuda.gamma_21").as_double();
        estimate_keypoints::iss3d_cuda::gamma_32 = node->get_parameter("estimate_keypoints.iss3d_cuda.gamma_32").as_double();
        estimate_keypoints::iss3d_cuda::score_threshold = node->get_parameter("estimate_keypoints.iss3d_cuda.score_threshold").as_double();

        estimate_keypoints::voxelisation::voxel_size = node->get_parameter("estimate_keypoints.voxelisation.voxel_size").as_double();

        // Descriptors parameters
        extract_feature_descriptors::method = node->get_parameter("extract_feature_descriptors.method").as_string();

        extract_feature_descriptors::fpfh::use_k_instead_of_radius = node->get_parameter("extract_feature_descriptors.fpfh.use_k_instead_of_radius").as_bool();
        extract_feature_descriptors::fpfh::fpfh_radius = node->get_parameter("extract_feature_descriptors.fpfh.fpfh_radius").as_double();
        extract_feature_descriptors::fpfh::fpfh_k = node->get_parameter("extract_feature_descriptors.fpfh.fpfh_k").as_int();

        extract_feature_descriptors::shot::shot_radius = node->get_parameter("extract_feature_descriptors.shot.shot_radius").as_double();
        extract_feature_descriptors::hpnvd::hpnvd_radius = node->get_parameter("extract_feature_descriptors.hpnvd.hpnvd_radius").as_double();
        extract_feature_descriptors::hpnvd::hpnvd_bin_num = node->get_parameter("extract_feature_descriptors.hpnvd.hpnvd_bin_num").as_int();

        // Matching parameters
        match_features::method = node->get_parameter("match_features.method").as_string();
        match_features::lowe_ratio = node->get_parameter("match_features.lowe_ratio").as_double();
        match_features::reciprocal = node->get_parameter("match_features.reciprocal").as_bool();
        match_features::sinkhorn::epsilon = node->get_parameter("match_features.Sinkhorn.epsilon").as_double();
        match_features::sinkhorn::dustbin_score = node->get_parameter("match_features.Sinkhorn.dustbin_score").as_double();
        match_features::sinkhorn::max_iterations = node->get_parameter("match_features.Sinkhorn.max_iterations").as_int();

        // Match rejection parameters
        match_rejection::inlier_threshold = node->get_parameter("match_rejection.inlier_threshold").as_double();
        match_rejection::max_iteration = node->get_parameter("match_rejection.max_iteration").as_int();

        // Transformation estimation parameters
        transformation_estimation::method = node->get_parameter("transformation_estimation.method").as_string();
        transformation_estimation::teaser::estimate_scaling = node->get_parameter("transformation_estimation.teaser.estimate_scaling").as_bool();
        transformation_estimation::teaser::noise_bound_multiplier = node->get_parameter("transformation_estimation.teaser.noise_bound_multiplier").as_double();
        transformation_estimation::teaser::rotation_estimation_algorithm = node->get_parameter("transformation_estimation.teaser.rotation_estimation_algorithm").as_string();
        transformation_estimation::teaser::rotation_gnc_factor = node->get_parameter("transformation_estimation.teaser.rotation_gnc_factor").as_double();
        transformation_estimation::teaser::rotation_max_iterations = node->get_parameter("transformation_estimation.teaser.rotation_max_iterations").as_int();
        transformation_estimation::teaser::rotation_cost_threshold = node->get_parameter("transformation_estimation.teaser.rotation_cost_threshold").as_double();
        transformation_estimation::teaser::inlier_selection_mode = node->get_parameter("transformation_estimation.teaser.inlier_selection_mode").as_string();
        transformation_estimation::teaser::inlier_graph_formulation = node->get_parameter("transformation_estimation.teaser.inlier_graph_formulation").as_string();
        // Candidate search parameters
        if (node->has_parameter("candidate_search.method")) candidate_search::method = node->get_parameter("candidate_search.method").as_string();
        if (node->has_parameter("candidate_search.max_distance")) candidate_search::max_distance = node->get_parameter("candidate_search.max_distance").as_double();
        if (node->has_parameter("candidate_search.temporal_exclusion")) candidate_search::temporal_exclusion = node->get_parameter("candidate_search.temporal_exclusion").as_int();
        if (node->has_parameter("candidate_search.ndtmc_dist_threshold")) candidate_search::ndtmc_dist_threshold = node->get_parameter("candidate_search.ndtmc_dist_threshold").as_double();
        if (node->has_parameter("candidate_search.ndtmc_resolution")) candidate_search::ndtmc_resolution = node->get_parameter("candidate_search.ndtmc_resolution").as_double();

        // Debug parameters
        debug::allow_pointcloud_extraction = node->get_parameter("debug.allow_pointcloud_extraction").as_bool();
    }

    inline void printconfig(rclcpp::Node* node) {
        if (!node) return;
        RCLCPP_INFO(node->get_logger(), "==============================================================================================");
        RCLCPP_INFO(node->get_logger(), "      PIPELINE CONFIGURATION LOADED :");
        RCLCPP_INFO(node->get_logger(), "----------------------------------------------------------------------------------------------");

        RCLCPP_INFO(node->get_logger(), "[GLOBAL/TEST]");
        RCLCPP_INFO(node->get_logger(), "  - Voxel size                                : %.3f", voxel_size);
        RCLCPP_INFO(node->get_logger(), "  - Sensor range                              : %.3f", sonar_range);

        RCLCPP_INFO(node->get_logger(), "[PIPELINE]");
        RCLCPP_INFO(node->get_logger(), "  - Normals calculations");
        RCLCPP_INFO(node->get_logger(), "      - Gaussian filter                       : %s", normal_calculation::gaussian_filter ? "TRUE" : "FALSE");
        RCLCPP_INFO(node->get_logger(), "      - Gaussian Width                        : %d", normal_calculation::gaussian_width);
        RCLCPP_INFO(node->get_logger(), "      - Gaussian Repetitions                  : %d", normal_calculation::gaussian_repetitions);
        RCLCPP_INFO(node->get_logger(), "      - Distance to TSDF                      : %.2f", normal_calculation::distance_to_tsdf);
        RCLCPP_INFO(node->get_logger(), "  - Keypoint Method                           : %s", estimate_keypoints::method.c_str());
        RCLCPP_INFO(node->get_logger(), "  - Keypoint Filtering Method                 : %s", estimate_keypoints::filtering_method.c_str());
        RCLCPP_INFO(node->get_logger(), "  - Keypoint Sampler Size (multiple)          : %.3f", estimate_keypoints::keypoint_sampler_size);
        RCLCPP_INFO(node->get_logger(), "  - Scale-Aware Keypoint Estimation           : %s (num_scales=%d, multiplier=%.2f)", 
                    estimate_keypoints::scale_aware::enabled ? "ENABLED" : "DISABLED",
                    estimate_keypoints::scale_aware::num_scales,
                    estimate_keypoints::scale_aware::scale_multiplier);
        RCLCPP_INFO(node->get_logger(), "  - 3D NMS Search Radius                      : %.3f m", estimate_keypoints::nms::radius);
        RCLCPP_INFO(node->get_logger(), "  - Fixed nb of keypoints                     : %s", estimate_keypoints::fixed_nb_of_kp ? "TRUE" : "FALSE");

        if (estimate_keypoints::method == "harris3d_cuda" || estimate_keypoints::method == "harris_cuda") {
            RCLCPP_INFO(node->get_logger(), "      - Score threshold                       : %.5f", estimate_keypoints::harris3d_cuda::score_threshold);
            RCLCPP_INFO(node->get_logger(), "      - k                                     : %.3f", estimate_keypoints::harris3d_cuda::k);
        }
        else if (estimate_keypoints::method == "iss3d_cuda" || estimate_keypoints::method == "iss_cuda") {
            RCLCPP_INFO(node->get_logger(), "      - Search Radius                         : %.3f", estimate_keypoints::iss3d_cuda::searchRadius);
            RCLCPP_INFO(node->get_logger(), "      - Gamma 21                              : %.3f", estimate_keypoints::iss3d_cuda::gamma_21);
            RCLCPP_INFO(node->get_logger(), "      - Gamma 32                              : %.3f", estimate_keypoints::iss3d_cuda::gamma_32);
            RCLCPP_INFO(node->get_logger(), "      - Score threshold                       : %.5f", estimate_keypoints::iss3d_cuda::score_threshold);
        }
        else if (estimate_keypoints::method == "harris3d" || estimate_keypoints::method == "harris") {
            RCLCPP_INFO(node->get_logger(), "      - Search Radius                         : %.3f", estimate_keypoints::harris3d::radius);
            RCLCPP_INFO(node->get_logger(), "      - Score threshold                       : %.5f", estimate_keypoints::harris3d::score_threshold);
        }
        else if (estimate_keypoints::method == "iss" || estimate_keypoints::method == "iss3d") {
            RCLCPP_INFO(node->get_logger(), "      - Search Radius                         : %.3f", estimate_keypoints::iss::searchRadius);
            RCLCPP_INFO(node->get_logger(), "      - Gamma 21                              : %.3f", estimate_keypoints::iss::gamma_21);
            RCLCPP_INFO(node->get_logger(), "      - Gamma 32                              : %.3f", estimate_keypoints::iss::gamma_32);
            RCLCPP_INFO(node->get_logger(), "      - Min Neighbors                         : %d", estimate_keypoints::iss::min_neighbors);
        }
        else if (estimate_keypoints::method == "voxelisation") {
            RCLCPP_INFO(node->get_logger(), "      - Voxel size                            : %.3f", estimate_keypoints::voxelisation::voxel_size);
        }

        RCLCPP_INFO(node->get_logger(), "  - Descriptor Method                         : %s", extract_feature_descriptors::method.c_str());
        if (extract_feature_descriptors::method == "fpfh") {
            RCLCPP_INFO(node->get_logger(), "      - Use k instead of radius               : %s", extract_feature_descriptors::fpfh::use_k_instead_of_radius ? "TRUE" : "FALSE");
            RCLCPP_INFO(node->get_logger(), "      - Radius                                : %.3f", extract_feature_descriptors::fpfh::fpfh_radius);
            RCLCPP_INFO(node->get_logger(), "      - k                                     : %d", extract_feature_descriptors::fpfh::fpfh_k);
        }
        else if (extract_feature_descriptors::method == "shot") {
            RCLCPP_INFO(node->get_logger(), "      - Radius                                : %.3f", extract_feature_descriptors::shot::shot_radius);
        }
        else if (extract_feature_descriptors::method == "hpnvd") {
            RCLCPP_INFO(node->get_logger(), "      - Radius                                : %.3f", extract_feature_descriptors::hpnvd::hpnvd_radius);
            RCLCPP_INFO(node->get_logger(), "      - Bin Num                               : %d", extract_feature_descriptors::hpnvd::hpnvd_bin_num);
        }
        RCLCPP_INFO(node->get_logger(), "  - Features matching Method                  : %s", match_features::method.c_str());
        RCLCPP_INFO(node->get_logger(), "      - Lowe's Ratio                          : %.2f", match_features::lowe_ratio);
        RCLCPP_INFO(node->get_logger(), "      - Reciprocal match                      : %s", match_features::reciprocal ? "TRUE" : "FALSE");
        if (match_features::method == "Sinkhorn") {
            RCLCPP_INFO(node->get_logger(), "      - Sinkhorn Epsilon                      : %.3f", match_features::sinkhorn::epsilon);
            RCLCPP_INFO(node->get_logger(), "      - Sinkhorn Dustbin Score                : %.3f", match_features::sinkhorn::dustbin_score);
            RCLCPP_INFO(node->get_logger(), "      - Sinkhorn Max Iterations               : %d", match_features::sinkhorn::max_iterations);
        }
        
        RCLCPP_INFO(node->get_logger(), "  - Transformation method                     : %s", transformation_estimation::method.c_str());
        
        if (transformation_estimation::method == "teaser") {
            RCLCPP_INFO(node->get_logger(), "  - TEASER++ :");
            RCLCPP_INFO(node->get_logger(), "      - Estimate scaling                      : %s", transformation_estimation::teaser::estimate_scaling ? "TRUE" : "FALSE");
            RCLCPP_INFO(node->get_logger(), "      - Noise bound multiplier                : %.2f", transformation_estimation::teaser::noise_bound_multiplier);
            RCLCPP_INFO(node->get_logger(), "      - Rotation algorithm                    : %s", transformation_estimation::teaser::rotation_estimation_algorithm.c_str());
            RCLCPP_INFO(node->get_logger(), "      - Rotation GNC factor                   : %.2f", transformation_estimation::teaser::rotation_gnc_factor);
            RCLCPP_INFO(node->get_logger(), "      - Rotation max iterations               : %d", transformation_estimation::teaser::rotation_max_iterations);
            RCLCPP_INFO(node->get_logger(), "      - Rotation cost threshold               : %.2e", transformation_estimation::teaser::rotation_cost_threshold);
            RCLCPP_INFO(node->get_logger(), "      - Inlier selection mode                 : %s", transformation_estimation::teaser::inlier_selection_mode.c_str());
            RCLCPP_INFO(node->get_logger(), "      - Inlier graph formulation              : %s", transformation_estimation::teaser::inlier_graph_formulation.c_str());
            RCLCPP_INFO(node->get_logger(), "      - Max clique time limit (in seconds)    : %.3f", transformation_estimation::teaser::max_clique_time_limit);
        }
        else if (transformation_estimation::method == "umeyama") {
            RCLCPP_INFO(node->get_logger(), "  - RANSAC :");
            RCLCPP_INFO(node->get_logger(), "      - Inlier Thresh                         : %.4f", match_rejection::inlier_threshold);
            RCLCPP_INFO(node->get_logger(), "      - Max Iter                              : %d", match_rejection::max_iteration);
        }
        
        RCLCPP_INFO(node->get_logger(), "[DEBUG]");
        RCLCPP_INFO(node->get_logger(), "  - Allow Pointcloud Extraction (impacts performance) : %s", debug::allow_pointcloud_extraction ? "TRUE" : "FALSE");
        
        RCLCPP_INFO(node->get_logger(), "========================================================");
    }
}