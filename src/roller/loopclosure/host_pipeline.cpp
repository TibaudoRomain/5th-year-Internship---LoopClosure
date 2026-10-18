#include <roller/loopclosure/host_pipeline.hpp>
#include <roller/base_node.hpp>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/io/pcd_io.h>

#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

using namespace roller::loopclosure;

loopclosure_master::loopclosure_master(rclcpp::Node *node){
    #ifdef NDEBUG
        std::cout << "[INFO] Program compiled in RELEASE mod (Optimisations enabled)" << std::endl;
    #else
        std::cout << "[INFO] Program compiled in DEBUG mod (Optimisations disabled)" << std::endl;
    #endif
    
    std::cout << "[Loop Closure Pipeline] : Started initializing" << std::endl;
    this->_node = node;
    this->_tree  = std::make_shared<pcl::search::KdTree<pcl::PointXYZI>>();
    
    this->_new_submap_keypoints_pub = _node->create_publisher<sensor_msgs::msg::PointCloud2>("/loopclosure/new_submap_keypoints", rclcpp::SensorDataQoS());
    this->_old_submap_keypoints_pub = _node->create_publisher<sensor_msgs::msg::PointCloud2>("/loopclosure/old_submap_keypoints", rclcpp::SensorDataQoS());
    this->_new_submap_pub = _node->create_publisher<sensor_msgs::msg::PointCloud2>("/loopclosure/new_submap_cloud", rclcpp::SensorDataQoS());
    this->_old_submap_pub = _node->create_publisher<sensor_msgs::msg::PointCloud2>("/loopclosure/old_submap_cloud", rclcpp::SensorDataQoS());
    this->_transformed_cloud_pub = _node->create_publisher<sensor_msgs::msg::PointCloud2>("/loopclosure/transformed_cloud", rclcpp::SensorDataQoS());
    this->_correspondences_pub = _node->create_publisher<visualization_msgs::msg::Marker>("/loopclosure/correspondences", rclcpp::SensorDataQoS());

    readconfig();
    this->config_tpp_params();
    
    
    this->cudacore = std::make_shared<CudaCore_Wrapper>(static_cast<float>(Config::voxel_size));
    /*this->cudacore->set_voxelSize(_voxel_size);
    this->cudacore->set_fpfhRadius(_fpfh_radius);
    this->cudacore->set_fpfh_k(_fpfh_k);
    this->cudacore->set_harris_nb_of_selected_points(_harris_nb_of_selected);
    this->cudacore->set_harris_threshold(_harris_outliers_threshold);
    this->cudacore->set_harris_fixed_nb_of_kp(_harris_fixed_nb_of_kp);
    this->cudacore->set_harris_score_threshold(_harris_score_threshold);
    this->cudacore->set_export_debug_harris_score(_export_debug_harris_score);*/
    std::cout << "[Loop Closure Pipeline] : Initialized" << std::endl;
}

void loopclosure_master::readconfig() {
    Config::readconfig(_node);
    Config::printconfig(_node);
}

void loopclosure_master::set_saving_directory(const std::string& saving_dir) {
    benchmark.set_saving_directory(saving_dir);
}

void loopclosure_master::LDPipeline_host(const Submap& submap, size_t submap_db_id, const std::vector<Keyframe>& submap_keyframes, const nanovdb::GridHandle<nanovdb::HostBuffer>& sdf){
    auto start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Adding a submap to the loop closure pipeline <-----------------" << std::endl;
    std::cout << "About to run submap throught CUDA ===========>>" << std::endl;  
    auto result = cudacore->LDPipeline_device(submap, submap_keyframes, sdf);
    std::cout << "Returned to master" << std::endl;
    references newsubmap;
    newsubmap.t_stamp = submap.poses.stamp;
    newsubmap.submap_db_id = submap_db_id;
    newsubmap.fg = submap.poses.fg;
    newsubmap.gt = submap.poses.gt;
    newsubmap.pose = submap.poses.fg.block<3, 1>(0, 3);

    newsubmap.keypoints = result.keypoints;
    newsubmap.descriptors = result.descriptors;         
    newsubmap.normals = result.normals;
    newsubmap.debug_harrisscore = result.debug_harrisscore;

    // Generate NDTMC global descriptor for candidate search
    NDTMC ndtmc_builder(static_cast<float>(Config::candidate_search::ndtmc_resolution));
    pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud = nullptr;

    if (submap.pointcloud_corrected && !submap.pointcloud_corrected->empty()) {
        target_cloud = submap.pointcloud_corrected;
    } else if (submap.pointcloud && !submap.pointcloud->empty()) {
        target_cloud = submap.pointcloud;
    } else if (result.keypoints && !result.keypoints->empty()) {
        target_cloud = result.keypoints;
    }

    if (target_cloud && !target_cloud->empty()) {
        newsubmap.ndtmc_descriptor = ndtmc_builder.createNDTMC(target_cloud);
        std::cout << "[NDTMC] Submap " << submap_db_id << " descriptor built ("
                  << newsubmap.ndtmc_descriptor.rows() << "x" << newsubmap.ndtmc_descriptor.cols() << ")" << std::endl;
    }

    bool no_keypoints = !result.keypoints || result.keypoints->empty();
    bool no_descriptor = std::visit([](const auto& cloud_ptr) { return !cloud_ptr || cloud_ptr->empty(); }, result.descriptors);

    if (!no_descriptor) {
        std::visit([this, &newsubmap](auto&& cloud_ptr) {
            using DescriptorT = typename std::decay_t<decltype(cloud_ptr)>::element_type::PointType;
            this->init_globalDescriptor<DescriptorT>(newsubmap);
        }, newsubmap.descriptors);
    }

    submap_storage.push_back(newsubmap);

    std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
    benchmark.log_description_time(duration.count());
    std::cout << "Submap described and added to pipeline in " << duration.count() << "ms" << std::endl;

    if (no_keypoints || no_descriptor) {
        std::cout << "[LoopClosure] Incomplete description for submap " << submap_db_id 
                  << " (keypoints: " << (result.keypoints ? result.keypoints->size() : 0)
                  << "). Skipping candidate matching for this submap." << std::endl;
        return;
    }
        
    if(submap_storage.size()==1){ 
        std::cout << "Only one Submap, to the next." << std::endl;
        return; 
    }

    std::cout << "Now " << submap_storage.size() << " Submaps described, looking for loopclosure candidates." << std::endl;
    
    auto start_search = std::chrono::high_resolution_clock::now();
    std::vector<references> candidates;
    candidate_search(submap_storage, newsubmap, candidates, 2.5f);
    double search_duration = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_search).count();
    benchmark.log_candidate_search_time(search_duration);

    for (references reference : candidates) {
        std::cout << "Comparing submap n°" << newsubmap.submap_db_id << " to submap n°" << reference.submap_db_id << std::endl;

        // Evaluate benchmark for this candidate pair
        double desc_dist = 0.0;
        int best_shift = 0;
        if (newsubmap.ndtmc_descriptor.rows() > 0 && reference.ndtmc_descriptor.rows() > 0) {
            auto dist_res = NDTMC::distanceBtnNDTScanContext(newsubmap.ndtmc_descriptor, reference.ndtmc_descriptor);
            if (!dist_res.empty()) {
                desc_dist = dist_res[0].first;
                best_shift = dist_res[0].second;
            }
        }

        double yaw_prior_deg = (best_shift >= 0) ? (best_shift * (360.0 / 60.0)) : 0.0;
        double yaw_prior_rad = yaw_prior_deg * (M_PI / 180.0);

        std::cout << "[NDTMC YAW PRIOR] Submap " << newsubmap.submap_db_id << " vs Submap " << reference.submap_db_id
                  << " -> Yaw Rotation Prior: " << std::fixed << std::setprecision(2) << yaw_prior_deg << " deg ("
                  << std::setprecision(4) << yaw_prior_rad << " rad, sector shift: " << best_shift << ")" << std::endl;

        CandidatePairEvaluation pair_eval = benchmark.evaluate_pair(
            newsubmap.submap_db_id, reference.submap_db_id,
            newsubmap.gt, reference.gt,
            desc_dist, best_shift, true
        );

        if (Config::transformation_estimation::method == "umeyama") {
            compare_submaps_umeyama(newsubmap, reference, &pair_eval);
        } else if (Config::transformation_estimation::method == "teaser") {
            compare_submaps_teaser(newsubmap, reference, &pair_eval);
        }
        debug(newsubmap, reference, _TransformationMatrix);
    }

    benchmark.print_submap_report(submap_db_id);
    benchmark.export_csv();
}


void loopclosure_master::candidate_search(const std::vector<references>& allsubmaps, const references& new_submap, std::vector<references>& candidates, float margin){
    std::cout << "[Candidate Search] Searching loop closure candidates for submap " << new_submap.submap_db_id 
              << " (method: " << Config::candidate_search::method << ")..." << std::endl;

    float max_dist = Config::sonar_range + margin;
    float max_dist_sq = max_dist * max_dist;
    Eigen::Vector3f npose = new_submap.pose;

    bool has_new_ndtmc = (new_submap.ndtmc_descriptor.rows() > 0 && new_submap.ndtmc_descriptor.cols() > 0);

    for (size_t idx = 0; idx < allsubmaps.size(); ++idx) {
        const auto& old_submap = allsubmaps[idx];

        // 1. Skip self-comparison
        if (old_submap.submap_db_id == new_submap.submap_db_id) {
            continue;
        }

        // 2. Skip immediate temporal neighbors (submaps close in time)
        if (new_submap.submap_db_id >= old_submap.submap_db_id && 
            (new_submap.submap_db_id - old_submap.submap_db_id) <= static_cast<size_t>(Config::candidate_search::temporal_exclusion)) {
            continue;
        }

        // 3. Coarse spatial bounding box / max distance check
        Eigen::Vector3f opose = old_submap.pose;
        float dx = (npose.x() - opose.x());
        float dy = (npose.y() - opose.y());
        float dz = (npose.z() - opose.z());
        float dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq > max_dist_sq) {
            continue;
        }

        // 4. NDTMC Descriptor Similarity Check
        bool has_old_ndtmc = (old_submap.ndtmc_descriptor.rows() > 0 && old_submap.ndtmc_descriptor.cols() > 0);

        if (Config::candidate_search::method == "ndtmc" && has_new_ndtmc && has_old_ndtmc) {
            Eigen::MatrixXd desc_new = new_submap.ndtmc_descriptor;
            Eigen::MatrixXd desc_old = old_submap.ndtmc_descriptor;

            auto dist_results = NDTMC::distanceBtnNDTScanContext(desc_new, desc_old);

            double min_sc_dist = 1e6;
            int best_yaw_shift = -1;

            for (const auto& res : dist_results) {
                if (res.first < min_sc_dist) {
                    min_sc_dist = res.first;
                    best_yaw_shift = res.second;
                }
            }

            double yaw_deg = (best_yaw_shift >= 0) ? (best_yaw_shift * (360.0 / 60.0)) : 0.0;
            double yaw_rad = yaw_deg * (M_PI / 180.0);

            std::cout << "[Candidate Search] Submap " << new_submap.submap_db_id 
                      << " vs Submap " << old_submap.submap_db_id 
                      << " -> NDTMC dist: " << std::fixed << std::setprecision(4) << min_sc_dist 
                      << " (yaw prior: " << std::setprecision(2) << yaw_deg << " deg / " 
                      << std::setprecision(4) << yaw_rad << " rad, shift: " << best_yaw_shift << ")" << std::endl;

            if (min_sc_dist <= Config::candidate_search::ndtmc_dist_threshold) {
                candidates.push_back(old_submap);
                std::cout << "  ==> ACCEPTED candidate: Submap n°" << old_submap.submap_db_id 
                          << " (dist=" << min_sc_dist << ", yaw_prior=" << yaw_deg << " deg)" << std::endl;
            } else {
                std::cout << "  --> Rejected candidate: Submap n°" << old_submap.submap_db_id 
                          << " (NDTMC dist " << min_sc_dist << " > threshold " << Config::candidate_search::ndtmc_dist_threshold << ")" << std::endl;
            }
        } else {
            // Fallback to spatial distance candidate search if NDTMC is disabled or missing
            candidates.push_back(old_submap);
            std::cout << "  ==> ACCEPTED candidate (distance fallback): Submap n°" << old_submap.submap_db_id 
                      << " (dist=" << std::sqrt(dist_sq) << "m)" << std::endl;
        }
    }
}


void loopclosure_master::compare_submaps_umeyama(const references& new_submap, const references& old_submap, CandidatePairEvaluation* eval){
    //----------------------------------------------MATCH FEATURES----------------------------------------------------------
    auto process_clock_start = std::chrono::high_resolution_clock::now();

    pcl::CorrespondencesPtr correspondences = std::make_shared<pcl::Correspondences>();
    std::visit([this, &correspondences](auto&& new_desc, auto&& old_desc) {
        using T1 = std::decay_t<decltype(new_desc)>;
        using T2 = std::decay_t<decltype(old_desc)>;
        
        // Ensure both submaps use the same descriptor type
        if constexpr (std::is_same_v<T1, T2>) {
            if (Config::match_features::method == "base_pcl") {
                this->match_features_pcl<typename T1::element_type::PointType>(new_desc, old_desc, *correspondences);
            } else if (Config::match_features::method == "Lowe_Ratio") {
                this->match_features_lr<typename T1::element_type::PointType>(new_desc, old_desc, *correspondences, Config::match_features::lowe_ratio);
            } else if (Config::match_features::method == "Sinkhorn") {
                this->match_features_sinkhorn<typename T1::element_type::PointType>(new_desc, old_desc, *correspondences, Config::match_features::sinkhorn::epsilon, Config::match_features::sinkhorn::dustbin_score, Config::match_features::sinkhorn::max_iterations);
            }
            RCLCPP_INFO(this->_node->get_logger(), "Success : Feature Matching, number of correspondences = %ld", correspondences->size());
        } else {
            RCLCPP_ERROR(this->_node->get_logger(), "Submaps have mismatched descriptor types!");
        }
    }, new_submap.descriptors, old_submap.descriptors);

    auto process_clock_match = std::chrono::high_resolution_clock::now();
    double match_time_ms = std::chrono::duration<double, std::milli>(process_clock_match - process_clock_start).count();
    RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", match_time_ms);

    //----------------------------------------------------------------------------------------------------------------------
    //---------------------------------------------REJECT MATCH-------------------------------------------------------------
    pcl::CorrespondencesPtr filtered_correspondences = std::make_shared<pcl::Correspondences>();
    
    this->match_rejection(correspondences, *filtered_correspondences, new_submap.normals, old_submap.normals);
    
    auto process_clock_rej = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO(this->_node->get_logger(), "Success : Feature Rejection, number of rejected = %ld", correspondences->size() - filtered_correspondences->size());

    const std::string frame_id = _node->has_parameter("world_frame") ? _node->get_parameter("world_frame").as_string() : "world";
    this->publish_correspondences(new_submap.keypoints, old_submap.keypoints, correspondences, *filtered_correspondences, frame_id, new_submap.t_stamp);

    //----------------------------------------------------------------------------------------------------------------------
    //-----------------------------------------ESTIMATE TRANSFORM-----------------------------------------------------------

    _TransformationMatrix = this->estimate_transform_umeyama(filtered_correspondences, new_submap.normals, old_submap.normals);

    auto process_clock_esti = std::chrono::high_resolution_clock::now();
    double estimate_time_ms = std::chrono::duration<double, std::milli>(process_clock_esti - process_clock_rej).count();
    bool success = (_TransformationMatrix != _error_matrix);

    if(success){
        RCLCPP_INFO(this->_node->get_logger(), "Success : Transformation estimation with Umeyama");
        RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", estimate_time_ms);
        RCLCPP_INFO(this->_node->get_logger(), "Transformation Matrix found : %s", Utils::matrix_to_string(_TransformationMatrix).c_str());
    }
    else{
        RCLCPP_INFO(this->_node->get_logger(), "Failure : Error Matrix detected");
    }

    if (eval) {
        benchmark.record_registration(
            *eval,
            success,
            correspondences->size(),
            filtered_correspondences->size(),
            _TransformationMatrix,
            match_time_ms,
            estimate_time_ms
        );
    }
}



void loopclosure_master::compare_submaps_teaser(const references& new_submap, const references& old_submap, CandidatePairEvaluation* eval){
    //----------------------------------------------MATCH FEATURES----------------------------------------------------------
    auto process_clock_start = std::chrono::high_resolution_clock::now();

    pcl::CorrespondencesPtr correspondences = std::make_shared<pcl::Correspondences>();
    std::visit([this, &correspondences](auto&& new_desc, auto&& old_desc) {
        using T1 = std::decay_t<decltype(new_desc)>;
        using T2 = std::decay_t<decltype(old_desc)>;
        
        // Ensure both submaps use the same descriptor type
        if constexpr (std::is_same_v<T1, T2>) {
            if (Config::match_features::method == "base_pcl") {
                this->match_features_pcl<typename T1::element_type::PointType>(new_desc, old_desc, *correspondences);
            } else if (Config::match_features::method == "Lowe_Ratio") {
                this->match_features_lr<typename T1::element_type::PointType>(new_desc, old_desc, *correspondences, Config::match_features::lowe_ratio);
            } else if (Config::match_features::method == "Sinkhorn") {
                this->match_features_sinkhorn<typename T1::element_type::PointType>(new_desc, old_desc, *correspondences, Config::match_features::sinkhorn::epsilon, Config::match_features::sinkhorn::dustbin_score, Config::match_features::sinkhorn::max_iterations);
            }
            RCLCPP_INFO(this->_node->get_logger(), "Success : Feature Matching, number of correspondences = %ld", correspondences->size());
        } else {
            RCLCPP_ERROR(this->_node->get_logger(), "Submaps have mismatched descriptor types!");
        }
    }, new_submap.descriptors, old_submap.descriptors);

    auto process_clock_match = std::chrono::high_resolution_clock::now();
    double match_time_ms = std::chrono::duration<double, std::milli>(process_clock_match - process_clock_start).count();
    RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", match_time_ms);

    //----------------------------------------------------------------------------------------------------------------------
    //---------------------------------------------TEASER ++-------------------------------------------------------------
    auto process_clock_tpp = std::chrono::high_resolution_clock::now();
    pcl::CorrespondencesPtr filtered_correspondences = std::make_shared<pcl::Correspondences>();
    
    _TransformationMatrix = this->estimate_transform_teaser(correspondences, *filtered_correspondences, new_submap.keypoints, old_submap.keypoints, Config::voxel_size);

    auto process_clock_esti = std::chrono::high_resolution_clock::now();
    double estimate_time_ms = std::chrono::duration<double, std::milli>(process_clock_esti - process_clock_tpp).count();
    bool success = (_TransformationMatrix != _error_matrix);

    if(success){
        RCLCPP_INFO(this->_node->get_logger(), "Success : Transformation estimation with Teaser ++");
        RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", estimate_time_ms);
        RCLCPP_INFO(this->_node->get_logger(), "Transformation Matrix found : %s", Utils::matrix_to_string(_TransformationMatrix).c_str());
    }
    else{
        RCLCPP_INFO(this->_node->get_logger(), "Failure : Error Matrix detected");
    }
    
    const std::string frame_id = _node->has_parameter("world_frame") ? _node->get_parameter("world_frame").as_string() : "world";
    this->publish_correspondences(new_submap.keypoints, old_submap.keypoints, correspondences, *filtered_correspondences, frame_id, new_submap.t_stamp);

    if (eval) {
        benchmark.record_registration(
            *eval,
            success,
            correspondences->size(),
            filtered_correspondences->size(),
            _TransformationMatrix,
            match_time_ms,
            estimate_time_ms
        );
    }
}

 
void loopclosure_master::match_rejection(pcl::CorrespondencesPtr correspondences, pcl::Correspondences& output, 
    pcl::PointCloud<pcl::PointNormal>::Ptr inputSource,  pcl::PointCloud<pcl::PointNormal>::Ptr inputTarget
){
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointNormal> ransac_rej;
    ransac_rej.setInputSource(inputSource);
    ransac_rej.setInputTarget(inputTarget);
    ransac_rej.setInputCorrespondences(correspondences);
    ransac_rej.setInlierThreshold(Config::match_rejection::inlier_threshold); 
    ransac_rej.setMaximumIterations(Config::match_rejection::max_iteration);
    pcl::Correspondences inliers;
    ransac_rej.getCorrespondences(output);
}

Eigen::Matrix4f loopclosure_master::estimate_transform_umeyama(
    pcl::CorrespondencesPtr correspondences,
    pcl::PointCloud<pcl::PointNormal>::Ptr inputSource,  
    pcl::PointCloud<pcl::PointNormal>::Ptr inputTarget
){
    if (correspondences->empty()) {
        RCLCPP_WARN(this->_node->get_logger(), "No correspondences provided to estimate_transform!");
        return _error_matrix;
    }

    int MatrixSize = 2* correspondences->size();

    Eigen::Matrix3Xf src(3, MatrixSize);
    Eigen::Matrix3Xf ref(3, MatrixSize);

    for (int i = 0; i < MatrixSize/2; ++i) {
        // Get the points and their normals
        int idx_query = correspondences->at(i).index_query;
        int idx_match = correspondences->at(i).index_match;
        
        const auto& input_point = inputSource->points[idx_query];
        const auto& reference_point = inputTarget->points[idx_match];

        Eigen::Vector3f input_pos = input_point.getVector3fMap();
        Eigen::Vector3f reference_pos = reference_point.getVector3fMap();

        Eigen::Vector3f input_normal = input_point.getNormalVector3fMap();
        Eigen::Vector3f reference_normal = reference_point.getNormalVector3fMap();

        input_normal.normalize();
        reference_normal.normalize();

        src.col(i) = input_pos;
        ref.col(i) = reference_pos;

        float weight = 0.1f;

        src.col(i + MatrixSize/2) = input_pos + (input_normal * weight);
        ref.col(i + MatrixSize/2) = reference_pos + (reference_normal * weight);
    }
    return Eigen::umeyama(src, ref, false);
}

Eigen::Matrix4f loopclosure_master::estimate_transform_teaser(
    pcl::CorrespondencesPtr correspondences,
    pcl::Correspondences& filtered_correspondences, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr sourceKeypoints,  
    pcl::PointCloud<pcl::PointXYZI>::Ptr targetKeypoints,
    double voxel_size)
{
    filtered_correspondences.clear();
    if (correspondences->size() < 3) {
        std::cerr << "[TEASER++] Pas assez de correspondances pour calculer une transformation." << std::endl;
        return _error_matrix;
    }
    size_t num_corrs = correspondences->size();
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points(3, num_corrs);
    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_points(3, num_corrs);

    for (size_t i = 0; i < num_corrs; ++i) {
        int src_idx = (*correspondences)[i].index_query;
        int tgt_idx = (*correspondences)[i].index_match;

        src_points(0, i) = static_cast<double>(sourceKeypoints->points[src_idx].x);
        src_points(1, i) = static_cast<double>(sourceKeypoints->points[src_idx].y);
        src_points(2, i) = static_cast<double>(sourceKeypoints->points[src_idx].z);

        tgt_points(0, i) = static_cast<double>(targetKeypoints->points[tgt_idx].x);
        tgt_points(1, i) = static_cast<double>(targetKeypoints->points[tgt_idx].y);
        tgt_points(2, i) = static_cast<double>(targetKeypoints->points[tgt_idx].z);
    }
    
    teaser::RobustRegistrationSolver::Params params = get_tpp_params();
    
    teaser::RobustRegistrationSolver solver(params);
    solver.solve(src_points, tgt_points);

    std::vector<int> inlier_indices = solver.getInlierMaxClique();
    filtered_correspondences.reserve(inlier_indices.size());
    for (int idx : inlier_indices) {
        filtered_correspondences.push_back((*correspondences)[idx]);
    }

    auto solution = solver.getSolution();
    Eigen::Matrix4f final_transform = Eigen::Matrix4f::Identity();

    if (solution.valid) {
        final_transform.block<3, 3>(0, 0) = solution.rotation.cast<float>();
        final_transform.block<3, 1>(0, 3) = solution.translation.cast<float>();
    } else {
        std::cerr << "[TEASER++] L'alignement a échoué (Solution invalide)." << std::endl;
        return _error_matrix;
    }

    return final_transform;
}


//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                              HELPERS
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

void loopclosure_master::config_tpp_params(){
    tpp_params.estimate_scaling = Config::transformation_estimation::teaser::estimate_scaling; 
    tpp_params.noise_bound = Config::voxel_size * Config::transformation_estimation::teaser::noise_bound_multiplier; 
    if(Config::transformation_estimation::teaser::rotation_estimation_algorithm == "GNC_TLS"){tpp_params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;}
    else if(Config::transformation_estimation::teaser::rotation_estimation_algorithm == "FGR"){tpp_params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;}
    else if(Config::transformation_estimation::teaser::rotation_estimation_algorithm == "QUATRO"){tpp_params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::QUATRO;}
    else{
        std::cerr << "[TEASER++] Unknown rotation algorithm, correct loopclosure.yaml." << std::endl;
    }
    tpp_params.rotation_gnc_factor = Config::transformation_estimation::teaser::rotation_gnc_factor;
    tpp_params.rotation_max_iterations = Config::transformation_estimation::teaser::rotation_max_iterations;
    tpp_params.rotation_cost_threshold = Config::transformation_estimation::teaser::rotation_cost_threshold;
    if(Config::transformation_estimation::teaser::inlier_selection_mode == "PMC_EXACT"){tpp_params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;}
    else if(Config::transformation_estimation::teaser::inlier_selection_mode == "PMC_HEU"){tpp_params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;}
    else if(Config::transformation_estimation::teaser::inlier_selection_mode == "KCORE_HEU"){tpp_params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::KCORE_HEU;}
    else if(Config::transformation_estimation::teaser::inlier_selection_mode == "NONE"){tpp_params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::NONE;}
    else{
        std::cerr << "[TEASER++] Unknown inlier selection mode, correct loopclosure.yaml." << std::endl;
    }
    if(Config::transformation_estimation::teaser::inlier_graph_formulation == "CHAIN"){tpp_params.rotation_tim_graph = teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::CHAIN;}
    else if(Config::transformation_estimation::teaser::inlier_graph_formulation == "COMPLETE"){tpp_params.rotation_tim_graph = teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::COMPLETE;}
    else{
        std::cerr << "[TEASER++] Unknown graph formulation, correct loopclosure.yaml." << std::endl;
    }
    tpp_params.max_clique_time_limit = Config::transformation_estimation::teaser::max_clique_time_limit;
}
teaser::RobustRegistrationSolver::Params loopclosure_master::get_tpp_params(){
    return tpp_params;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                              PUBLISHERS
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

void loopclosure_master::debug(const references& new_submap, const references& old_submap, const Eigen::Matrix4f& transformation){
    pcl::PointCloud<pcl::PointNormal>::Ptr transformed_cloud = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    pcl::transformPointCloudWithNormals(*new_submap.normals, *transformed_cloud, transformation);

    const std::string frame_id = _node->has_parameter("world_frame")
        ? _node->get_parameter("world_frame").as_string()
        : std::string("world");

    if (new_submap.keypoints) {
        sensor_msgs::msg::PointCloud2 new_keypoints_msg;
        pcl::toROSMsg(*new_submap.keypoints, new_keypoints_msg);
        new_keypoints_msg.header.frame_id = frame_id;
        new_keypoints_msg.header.stamp = new_submap.t_stamp;
        _new_submap_keypoints_pub->publish(new_keypoints_msg);
    }

    if (old_submap.keypoints) {
        sensor_msgs::msg::PointCloud2 old_keypoints_msg;
        pcl::toROSMsg(*old_submap.keypoints, old_keypoints_msg);
        old_keypoints_msg.header.frame_id = frame_id;
        old_keypoints_msg.header.stamp = old_submap.t_stamp;
        _old_submap_keypoints_pub->publish(old_keypoints_msg);
    }

    this->publish_debug_cloud(new_submap.normals, _new_submap_pub, frame_id, new_submap.t_stamp);
    this->publish_debug_cloud(old_submap.normals, _old_submap_pub, frame_id, old_submap.t_stamp);
    this->publish_debug_cloud(transformed_cloud, _transformed_cloud_pub, frame_id, new_submap.t_stamp);
}


void loopclosure_master::publish_debug_cloud(const pcl::PointCloud<pcl::PointNormal>::ConstPtr& cloud,
                                             const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher,
                                             const std::string& frame_id,
                                             const rclcpp::Time& stamp){
    if (!publisher || !cloud) {
        return;
    }

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = frame_id;
    cloud_msg.header.stamp = stamp;
    publisher->publish(cloud_msg);
}

void loopclosure_master::publish_correspondences(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source_keypoints,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& target_keypoints,
    const pcl::CorrespondencesPtr& all_correspondences,
    const pcl::Correspondences& filtered_correspondences,
    const std::string& frame_id,
    const rclcpp::Time& stamp)
{
    if (!_correspondences_pub || all_correspondences->empty() || !source_keypoints || !target_keypoints) return;

    visualization_msgs::msg::Marker green_marker;
    green_marker.header.frame_id = frame_id;
    green_marker.header.stamp = stamp;
    green_marker.ns = "validated_correspondences";
    green_marker.id = 0;
    green_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    green_marker.action = visualization_msgs::msg::Marker::ADD;

    visualization_msgs::msg::Marker red_marker;
    red_marker.header.frame_id = frame_id;
    red_marker.header.stamp = stamp;
    red_marker.ns = "eliminated_correspondences";
    red_marker.id = 1;
    red_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    red_marker.action = visualization_msgs::msg::Marker::ADD;
    
    green_marker.scale.x = 0.04;
    red_marker.scale.x = 0.03; 
    
    green_marker.color.r = 0.0f;
    green_marker.color.g = 1.0f;
    green_marker.color.b = 0.0f;
    green_marker.color.a = 1.0f;
    
    red_marker.color.r = 1.0f;
    red_marker.color.g = 0.0f;
    red_marker.color.b = 0.0f;
    red_marker.color.a = 0.5f; 

    std::set<std::pair<int, int>> inlier_set;
    for (const auto& corr : filtered_correspondences) {
        inlier_set.insert({corr.index_query, corr.index_match});
    }

    for (const auto& corr : *all_correspondences) {
        geometry_msgs::msg::Point p1, p2;
        
        p1.x = source_keypoints->points[corr.index_query].x;
        p1.y = source_keypoints->points[corr.index_query].y;
        p1.z = source_keypoints->points[corr.index_query].z;
        
        p2.x = target_keypoints->points[corr.index_match].x;
        p2.y = target_keypoints->points[corr.index_match].y;
        p2.z = target_keypoints->points[corr.index_match].z;
        
        if (inlier_set.find({corr.index_query, corr.index_match}) != inlier_set.end()) {
            green_marker.points.push_back(p1);
            green_marker.points.push_back(p2);
        } 
        else {
            red_marker.points.push_back(p1);
            red_marker.points.push_back(p2);
        }
    }

    _correspondences_pub->publish(green_marker);
    _correspondences_pub->publish(red_marker);
}




//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                              DOWNLOADS
//-------------------------------------------------------------------------------------------------------------------------------------------------------------


void loopclosure_master::download_submaps_description(){
    // Generate timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();
    
    // Create base directory
    std::string base_dir = "/tmp/roller/submap_descriptions/" + timestamp + "/";
    std::filesystem::create_directories(base_dir);
    
    std::cout << "[LoopClosure] Downloading submap descriptions to: " << base_dir << std::endl;
    if (!Config::debug::allow_pointcloud_extraction) {
        std::cout << "[LoopClosure] WARNING: debug.allow_pointcloud_extraction is FALSE in config. Surface normals and debug score pointclouds were not extracted to optimize performance." << std::endl;
    }
    
    // Download keypoints, normals, and debug score clouds for each submap
    for(const auto& submap : submap_storage){
        std::string id_str = std::to_string(submap.submap_db_id);
        
        // Save keypoints
        if(submap.keypoints && !submap.keypoints->empty()){
            std::string keypoints_file = base_dir + "submap_" + id_str + "_keypoints.pcd";
            pcl::io::savePCDFile(keypoints_file, *submap.keypoints, true);
            std::cout << "[LoopClosure] Saved keypoints to: " << keypoints_file << std::endl;
        }
        
        // Save normals
        if(submap.normals && !submap.normals->empty()){
            std::string normals_file = base_dir + "submap_" + id_str + "_normals.pcd";
            pcl::io::savePCDFile(normals_file, *submap.normals, true);
            std::cout << "[LoopClosure] Saved normals to: " << normals_file << std::endl;
        }
        
        // Save debug harris score cloud if available
        if(submap.debug_harrisscore && !submap.debug_harrisscore->empty()){
            std::string harris_debug_file = base_dir + "submap_" + id_str + "_debug_score.pcd";
            pcl::io::savePCDFile(harris_debug_file, *submap.debug_harrisscore, true);
            std::cout << "[LoopClosure] Saved debug score to: " << harris_debug_file << std::endl;
        }
    }
    
    std::cout << "[LoopClosure] Submap descriptions download complete" << std::endl;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                              CLEANING
//-------------------------------------------------------------------------------------------------------------------------------------------------------------
