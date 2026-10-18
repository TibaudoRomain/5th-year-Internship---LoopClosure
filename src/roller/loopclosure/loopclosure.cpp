#include <roller/loopclosure/loopclosure.hpp>
#include <roller/base_node.hpp>
#include <utility>

#include <unordered_map>
#include <vector>

namespace flann {
namespace serialization {
    template <typename Archive>
    void serialize(Archive&, std::unordered_map<unsigned int, std::vector<unsigned int>>&) {}
}
}

#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>

#include <pcl/features/fpfh.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/rops_estimation.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <pcl/point_struct_traits.h>

#define PCL_NO_PRECOMPILE
POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::Histogram<135>,
    (float, histogram, histogram)
)



#define METHOD_SUBSAMPLING 0
#define METHOD_VOXELISATION 1
#define METHOD_ISS 2
#define METHOD_HARRIS 3
#define METHOD_SIFT 4
#define METHOD_TRAJKOVIC 5


using namespace roller::loopclosure;


loop_closure::loop_closure(Base_node *node){
    this->_node = node;
    this->_tree  = std::make_shared<pcl::search::KdTree<pcl::PointXYZI>>();

    readconfig();

    _src_input_cloud = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    _src_reference_cloud =std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    _reference_cloud = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    _input_cloud =std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    _transformed_cloud =std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
}

void loop_closure::readconfig(){
    std::string pkg_path = ament_index_cpp::get_package_share_directory("roller");

    _voxel_size = _node->get_parameter("voxel_size").as_double();

    _is_a_test = _node->get_parameter("test_parameters.is_a_test").as_bool();
    _number_of_iterations = _node->get_parameter("test_parameters.iterations_nb").as_int();
    _extraction_method = _node->get_parameter("pipeline.estimate_keypoints.extraction_method").as_int();
    _extraction_strength = _node->get_parameter("pipeline.estimate_keypoints.strength").as_double();
    _fpfh_radius = _node->get_parameter("pipeline.extract_feature_descriptors.fpfh_radius").as_double();
    _reciprocal_match = _node->get_parameter("pipeline.match_features.reciprocal").as_bool();
    _ransac_inlier_threshold = _node->get_parameter("pipeline.match_rejection.inlier_threshold").as_double();
    _ransac_max_iter = _node->get_parameter("pipeline.match_rejection.max_iteration").as_int();

    _use_external_as_input = _node->get_parameter("input_parameters.use_external_pointcloud").as_bool();
    _path_to_input = pkg_path + _node->get_parameter("input_parameters.path_to_normals").as_string();
    _use_external_as_reference = _node->get_parameter("reference_parameters.use_external_pointcloud").as_bool();
    _path_to_reference = pkg_path + _node->get_parameter("reference_parameters.path_to_normals").as_string();    
    _add_noise = _node->get_parameter("reference_parameters.add_noise").as_bool();
    _noise_intensity = _node->get_parameter("reference_parameters.noise_intensity").as_double();

    /*
    RCLCPP_INFO(_node->get_logger(), "========================================================");
    RCLCPP_INFO(_node->get_logger(), "PIPELINE CONFIGURATION LOADED :");
    RCLCPP_INFO(_node->get_logger(), "--------------------------------------------------------");

    // Section Globale & Test
    RCLCPP_INFO(_node->get_logger(), "[GLOBAL/TEST]");
    RCLCPP_INFO(_node->get_logger(), "  - Voxel size            : %.3f", _voxel_size);
    RCLCPP_INFO(_node->get_logger(), "  - Is a test             : %s", _is_a_test ? "TRUE" : "FALSE");
    RCLCPP_INFO(_node->get_logger(), "  - Number of iterations  : %d", _number_of_iterations);

    // Section Pipeline
    RCLCPP_INFO(_node->get_logger(), "[PIPELINE]");
    RCLCPP_INFO(_node->get_logger(), "  - Extraction method     : %d", _extraction_method);
    RCLCPP_INFO(_node->get_logger(), "  - Extraction strength   : %.3f", _extraction_strength);
    RCLCPP_INFO(_node->get_logger(), "  - FPFH Radius           : %.3f", _fpfh_radius);
    RCLCPP_INFO(_node->get_logger(), "  - Reciprocal match      : %s", _reciprocal_match ? "TRUE" : "FALSE");
    RCLCPP_INFO(_node->get_logger(), "  - RANSAC Inlier Thresh  : %.4f", _ransac_inlier_threshold);
    RCLCPP_INFO(_node->get_logger(), "  - RANSAC Max Iter       : %d", _ransac_max_iter);

    // Section Input
    RCLCPP_INFO(_node->get_logger(), "[INPUT]");
    RCLCPP_INFO(_node->get_logger(), "  - Use external input    : %s", _use_external_as_input ? "TRUE" : "FALSE");
    RCLCPP_INFO(_node->get_logger(), "  - Path to input         : %s", _path_to_input.c_str());

    // Section Reference
    RCLCPP_INFO(_node->get_logger(), "[REFERENCE]");
    RCLCPP_INFO(_node->get_logger(), "  - Use external ref      : %s", _use_external_as_reference ? "TRUE" : "FALSE");
    RCLCPP_INFO(_node->get_logger(), "  - Path to reference     : %s", _path_to_reference.c_str());
    RCLCPP_INFO(_node->get_logger(), "  - Add noise             : %s", _add_noise ? "TRUE" : "FALSE");
    if (_add_noise) {
        RCLCPP_INFO(_node->get_logger(), "  - Noise intensity       : %.4f", _noise_intensity);
    }

    RCLCPP_INFO(_node->get_logger(), "========================================================");*/
}

void loop_closure::get_pointcloud(){
    if(_use_external_as_input){
        if (pcl::io::loadPCDFile<pcl::PointNormal>(_path_to_input, *_src_input_cloud) == -1) {
            RCLCPP_ERROR(this->_node->get_logger(), "Couldn't read input file: %s", _path_to_input.c_str());
            _files_ok = false;
        }
    }
    if(_use_external_as_reference){
        if (pcl::io::loadPCDFile<pcl::PointNormal>(_path_to_reference, *_src_reference_cloud) == -1) {
            RCLCPP_ERROR(this->_node->get_logger(), "Couldn't read reference file: %s", _path_to_reference.c_str());
            _files_ok = false;
        }
    }

    if(!_files_ok)return;

    RCLCPP_INFO(this->_node->get_logger(), "Loaded %zu points for input and %zu for target.",
                 _src_input_cloud->size(), _src_reference_cloud->size());
                
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*_src_input_cloud, *_src_input_cloud, indices);
    pcl::removeNaNFromPointCloud(*_src_reference_cloud, *_src_reference_cloud, indices);

    _input_centroid = Utils::get_centroid(*_src_input_cloud);
    _reference_centroid = Utils::get_centroid(*_src_reference_cloud);

    //Noise and other modifications to the pointcloud
    if(_add_noise){
        Noise::addGaussianNoise(*_src_reference_cloud, 0.01f);
    }
}


void loop_closure::center_pointclouds(){
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translate(-_reference_centroid.head<3>());
    pcl::transformPointCloudWithNormals(*_src_reference_cloud, *_src_reference_cloud, transform);
    pcl::transformPointCloudWithNormals(*_src_input_cloud, *_src_input_cloud, transform);
}


void loop_closure::move_reference(float x_angle, float y_angle, float z_angle, float x_translation, float y_translation, float z_translation){
    float x_rad_angle = x_angle * M_PI / 180.0f;
    float y_rad_angle = y_angle * M_PI / 180.0f;
    float z_rad_angle = z_angle * M_PI / 180.0f;
    
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        
    transform.translate(Eigen::Vector3f(x_translation, y_translation, z_translation));

    transform.rotate(Eigen::AngleAxisf(x_rad_angle, Eigen::Vector3f::UnitX()));
    transform.rotate(Eigen::AngleAxisf(y_rad_angle, Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf(z_rad_angle, Eigen::Vector3f::UnitZ()));

    _modif_transform = transform.matrix();

    pcl::transformPointCloudWithNormals(*_src_reference_cloud, *_reference_cloud, _modif_transform);
}

/*
auto start_time = this->_node->now();

auto end_time = this->_node->now();
double duration = (end_time - start_time).seconds();
RCLCPP_INFO(this->_node->get_logger(), 
            "Timer ended in : %.6f secondes (Méthode: %d)", duration);
*/

pcl::PointCloud<pcl::PointXYZI>::Ptr loop_closure::estimate_keypoints(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, int method, float strength) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr return_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    switch (method) {
        case METHOD_ISS: {
            pcl::ISSKeypoint3D<pcl::PointNormal, pcl::PointXYZI> iss;
            iss.setInputCloud(cloud);
            iss.setThreshold21(0.975);
            iss.setThreshold32(0.975);
            iss.setSalientRadius(strength); // À ajuster selon l'échelle sonar
            iss.setNonMaxRadius(0.3);
            iss.compute(*return_cloud);
            break;
        }
        case METHOD_HARRIS: {           
            float leaf_size = 2 * _voxel_size;
            
            
            /*pcl::PointCloud<pcl::PointNormal>::Ptr search_surface(new pcl::PointCloud<pcl::PointNormal>);
            pcl::VoxelGrid<pcl::PointNormal> vox;
            vox.setInputCloud(cloud);
            vox.setLeafSize(leaf_size, leaf_size, leaf_size);
            vox.filter(*search_surface);*/  
                   
            
            pcl::HarrisKeypoint3D<pcl::PointNormal, pcl::PointXYZI, pcl::PointNormal> harris;
            harris.setInputCloud(cloud);
            harris.setNormals(cloud);

            pcl::search::KdTree<pcl::PointNormal>::Ptr temp_tree(new pcl::search::KdTree<pcl::PointNormal>());
            harris.setSearchMethod(temp_tree);

            
            harris.setRadius(strength);
            harris.setThreshold(0.001f);
            harris.setNonMaxSupression(true);
            harris.setRefine(false);
            harris.setNumberOfThreads(0);
            
            
            harris.compute(*return_cloud);
            break;
        }
        default:
            RCLCPP_ERROR(this->_node->get_logger(), "Méthode d'extraction inconnue !");
            pcl::copyPointCloud(*cloud, *return_cloud);
            break;
    }
    return return_cloud;
}



pcl::PointCloud<pcl::FPFHSignature33>::Ptr loop_closure::extract_fpfh_descriptors(pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints, pcl::PointCloud<pcl::PointNormal>::Ptr full_cloud){
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr return_features(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::FPFHEstimation<pcl::PointXYZI, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
    
    fpfh.setInputCloud(keypoints);
    if(_extraction_method != 0 && _extraction_method != 1){
        pcl::PointCloud<pcl::PointXYZI>::Ptr search_area(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::copyPointCloud(*full_cloud, *search_area);
        fpfh.setSearchSurface(search_area);    
    }

    fpfh.setInputNormals(full_cloud);
    fpfh.setSearchMethod(_tree);
    fpfh.setRadiusSearch(_fpfh_radius);

    //fpfh.setNumberOfThreads(0);
    fpfh.compute(*return_features);

    for (auto& point : return_features->points) {
        if (!std::isfinite(point.histogram[0])) {
            // Remplit l'histogramme de 0 si NaN détecté
            std::fill(std::begin(point.histogram), std::end(point.histogram), 0.0f);
        }
    }

    return return_features;
}

pcl::PointCloud<RoPSSignature>::Ptr loop_closure::extract_rops_descriptors(
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints, 
    pcl::PointCloud<pcl::PointNormal>::Ptr full_cloud)
{
    pcl::PointCloud<RoPSSignature>::Ptr return_features(new pcl::PointCloud<RoPSSignature>());
    
    // Sécurité anticrash si les nuages sont vides
    if (!keypoints || keypoints->empty() || !full_cloud || full_cloud->empty()) {
        return return_features;
    }

    float support_radius = 0.0285f;
    unsigned int number_of_bins = 5;
    unsigned int number_of_rotations = 3;
    
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree_xyzi(new pcl::search::KdTree<pcl::PointXYZI>());
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree_normal(new pcl::search::KdTree<pcl::PointNormal>());
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    // 1. Triangulation sur le nuage complet
    gp3.setSearchRadius(2.0 * support_radius);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4);
    gp3.setMinimumAngle(M_PI / 18);
    gp3.setMaximumAngle(2 * M_PI / 3);
    gp3.setNormalConsistency(false);
    gp3.setInputCloud(full_cloud);
    gp3.setSearchMethod(tree_normal);
    gp3.reconstruct(triangles);

    // 2. Préparation du nuage de surface au format XYZI (Requis par RoPS)
    pcl::PointCloud<pcl::PointXYZI>::Ptr full_cloud_xyzi(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::copyPointCloud(*full_cloud, *full_cloud_xyzi);

    // 3. Extraction des indices des keypoints par rapport au nuage complet
    // Pour éviter le crash d'indexation, RoPS doit chercher ses voisins dans un KdTree 
    // qui correspond parfaitement aux indices de la triangulation.
    pcl::search::KdTree<pcl::PointXYZI> keypoint_tree;
    keypoint_tree.setInputCloud(full_cloud_xyzi);
    
    pcl::IndicesPtr keypoints_indices(new std::vector<int>());
    keypoints_indices->reserve(keypoints->size());

    for (const auto& kp : keypoints->points) {
        std::vector<int> k_indices(1);
        std::vector<float> k_sqr_distances(1);
        if (keypoint_tree.nearestKSearch(kp, 1, k_indices, k_sqr_distances) > 0) {
            // On ne garde le point que s'il est vraiment très proche (éviter les faux résidus)
            if (k_sqr_distances[0] < 0.001f) { 
                keypoints_indices->push_back(k_indices[0]);
            }
        }
    }

    // 4. Configuration de RoPS
    pcl::ROPSEstimation<pcl::PointXYZI, RoPSSignature> rops;
    
    // Crucial : InputCloud = Le nuage complet (cohérent avec les triangles !)
    rops.setInputCloud(full_cloud_xyzi);
    rops.setSearchMethod(tree_xyzi);
    
    // Crucial : On restreint le calcul aux indices des Keypoints
    rops.setIndices(keypoints_indices);
    
    rops.setRadiusSearch(support_radius);
    rops.setTriangles(triangles.polygons);

    rops.setNumberOfPartitionBins(number_of_bins);
    rops.setNumberOfRotations(number_of_rotations);
    rops.setSupportRadius(support_radius);

    // 5. Calcul des descripteurs
    rops.compute(*return_features);

    // 6. Nettoyage des NaN
    for (auto& point : return_features->points) {
        if (!std::isfinite(point.histogram[0])) {
            std::fill(std::begin(point.histogram), std::end(point.histogram), 0.0f);
        }
    }

    return return_features;
}


void loop_closure::match_features(
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr input, 
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr reference)
{

    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> est;
    _correspondences = std::make_shared<pcl::Correspondences>();
    est.setInputSource (input);
    est.setInputTarget (reference);
    if(_reciprocal_match){est.determineReciprocalCorrespondences(*_correspondences);}
    else{est.determineCorrespondences(*_correspondences);}
}

void loop_closure::match_features(
    pcl::PointCloud<RoPSSignature>::Ptr input, 
    pcl::PointCloud<RoPSSignature>::Ptr reference)
{
    _correspondences = std::make_shared<pcl::Correspondences>();
    
    if (input->empty() || reference->empty()) return;

    // 1. Cas Correspondance Réciproque : On cherche dans les deux sens
    if (_reciprocal_match) 
    {
        pcl::KdTreeFLANN<RoPSSignature> kdtree_target;
        pcl::KdTreeFLANN<RoPSSignature> kdtree_source;
        kdtree_target.setInputCloud(reference);
        kdtree_source.setInputCloud(input);

        for (size_t i = 0; i < input->size(); ++i)
        {
            std::vector<int> idx_t(1); std::vector<float> dist_t(1);
            if (kdtree_target.nearestKSearch(input->points[i], 1, idx_t, dist_t) > 0)
            {
                std::vector<int> idx_s(1); std::vector<float> dist_s(1);
                if (kdtree_source.nearestKSearch(reference->points[idx_t[0]], 1, idx_s, dist_s) > 0)
                {
                    // Si le plus proche de la cible repointe bien sur la source (Réciproque)
                    if (idx_s[0] == static_cast<int>(i))
                    {
                        pcl::Correspondence corr(i, idx_t[0], dist_t[0]);
                        _correspondences->push_back(corr);
                    }
                }
            }
        }
    }
    // 2. Cas Correspondance Standard (avec Ratio Test pour filtrer le bruit)
    else 
    {
        pcl::KdTreeFLANN<RoPSSignature> kdtree;
        kdtree.setInputCloud(reference);

        const float max_ratio = 0.8f; // Seuil classique (ajustable entre 0.7 et 0.8)

        for (size_t i = 0; i < input->size(); ++i)
        {
            std::vector<int> indices(2);
            std::vector<float> sqr_distances(2);

            // On cherche les 2 plus proches voisins
            if (kdtree.nearestKSearch(input->points[i], 2, indices, sqr_distances) > 0)
            {
                // Test du ratio : le premier voisin doit être significativement plus proche que le second
                if (sqr_distances[0] < max_ratio * sqr_distances[1])
                {
                    pcl::Correspondence corr(i, indices[0], sqr_distances[0]);
                    _correspondences->push_back(corr);
                }
            }
        }
    }
}

void loop_closure::match_rejection(pcl::CorrespondencesPtr correspondences){
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointNormal> ransac_rej;
    _filtered_correspondences = std::make_shared<pcl::Correspondences>();
    ransac_rej.setInputSource(_input_cloud);
    ransac_rej.setInputTarget(_reference_cloud);
    ransac_rej.setInputCorrespondences(correspondences);
    ransac_rej.setInlierThreshold(_ransac_inlier_threshold); 
    ransac_rej.setMaximumIterations(_ransac_max_iter);
    pcl::Correspondences inliers;
    ransac_rej.getCorrespondences(inliers);
    *_filtered_correspondences = inliers;
}

Eigen::Matrix4f loop_closure::estimate_transform(pcl::CorrespondencesPtr correspondences){
    if (correspondences->empty()) {
        RCLCPP_WARN(this->_node->get_logger(), "No correspondences provided to estimate_transform!");
        return Eigen::Matrix4f::Identity();
    }

    int MatrixSize = 2* correspondences->size();

    Eigen::Matrix3Xf src(3, MatrixSize);
    Eigen::Matrix3Xf ref(3, MatrixSize);

    for (int i = 0; i < MatrixSize/2; ++i) {
        // Get the points and their normals
        int idx_query = correspondences->at(i).index_query;
        int idx_match = correspondences->at(i).index_match;
        
        const auto& input_point = _input_cloud->points[idx_query];
        const auto& reference_point = _reference_cloud->points[idx_match];

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



void loop_closure::on_start_pipeline(){
    RCLCPP_INFO(this->_node->get_logger(), "===== Pipeline Reception=====");
    
    get_pointcloud();                       //put point clouds into _src_..._cloud
    if(!_files_ok){ return;}
        
    //Pointcloud preparation step
    
    RCLCPP_INFO(this->_node->get_logger(), "Success : PointCloud Aquisition");
    center_pointclouds();                   //center _src_..._cloud
    RCLCPP_INFO(this->_node->get_logger(), "Success : PointClouds Centered");
    
    float translation_eval = 0;
    float rotation_eval = 0;

    auto start_time = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO(this->_node->get_logger(), "Timer Started");
    
    for(int i = 0; i<_number_of_iterations;i++){
        
        
        
        auto process_clock_start = std::chrono::high_resolution_clock::now();
        move_reference( Utils::getRandom(0.0f,360.0f), 
                        Utils::getRandom(0.0f,360.0f), 
                        Utils::getRandom(0.0f,360.0f), 
                        Utils::getRandom(0.0f,25.0f), 
                        Utils::getRandom(0.0f,25.0f), 
                        Utils::getRandom(0.0f,25.0f)); //give values to _reference_cloud
        pcl::copyPointCloud(*_src_input_cloud, *_input_cloud);
        
        RCLCPP_INFO(this->_node->get_logger(), "Success : Reference moved");
        auto process_clock_mr = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> process_duration = process_clock_mr - process_clock_start;
        RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", process_duration.count());
        
        //----------------------------------------------------------------------------------------------------------------------
        //-------------------------------------------ESTIMATE KEYPOINTS---------------------------------------------------------
        //Pipeline Calculations        
        _input_keypoints = this->estimate_keypoints(_input_cloud, _extraction_method, _extraction_strength);
        RCLCPP_INFO(this->_node->get_logger(), "Success : Estimating Input Keypoints, number of keypoints = %ld ", _input_keypoints->size());

        auto process_clock_ek_i = std::chrono::high_resolution_clock::now();
        process_duration = process_clock_ek_i - process_clock_mr;
        RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", process_duration.count());
        
        //----------------------------------------------------------------------------------------------------------------------

        _reference_keypoints = this->estimate_keypoints(_reference_cloud, _extraction_method, _extraction_strength);
        RCLCPP_INFO(this->_node->get_logger(), "Success : Estimating Reference Keypoints, number of keypoints = %ld ", _reference_keypoints->size());

        auto process_clock_ek_r = std::chrono::high_resolution_clock::now();
        process_duration = process_clock_ek_r - process_clock_ek_i;
        RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", process_duration.count());

        //----------------------------------------------------------------------------------------------------------------------
        //----------------------------------------EXTRACT FEATURE DESCRIPTORS---------------------------------------------------

        auto input_extracted = this->extract_rops_descriptors(_input_keypoints, _input_cloud); //input_extracted need to be downloaded from cuda
        RCLCPP_INFO(this->_node->get_logger(), "Success : Feature Extracted, number of features = %ld", input_extracted->size());

        auto process_clock_ex_i = std::chrono::high_resolution_clock::now();
        process_duration = process_clock_ex_i - process_clock_ek_r;
        RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", process_duration.count());

        //----------------------------------------------------------------------------------------------------------------------

        auto reference_extracted = this->extract_rops_descriptors(_reference_keypoints, _reference_cloud); //reference_extracted need to be downloaded from cuda

        auto process_clock_ex_r = std::chrono::high_resolution_clock::now();
        process_duration = process_clock_ex_r - process_clock_ex_i;
        RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", process_duration.count());

        //----------------------------------------------------------------------------------------------------------------------
        //------------------------------------------------END OF CUDA-----------------------------------------------------------
        //----------------------------------------------MATCH FEATURES----------------------------------------------------------

        this->match_features( input_extracted, reference_extracted);
        RCLCPP_INFO(this->_node->get_logger(), "Success : Feature Matching, number of correspondences = %ld", _correspondences->size());

        auto process_clock_match = std::chrono::high_resolution_clock::now();
        process_duration = process_clock_match - process_clock_ex_r;
        RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", process_duration.count());

        //----------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------REJECT MATCH-------------------------------------------------------------
        
        this->match_rejection(_correspondences);
        RCLCPP_INFO(this->_node->get_logger(), "Success : Feature Rejection, number of rejected = %ld", _correspondences->size() - _filtered_correspondences->size());

        auto process_clock_rej = std::chrono::high_resolution_clock::now();
        process_duration = process_clock_rej - process_clock_match;
        RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", process_duration.count());

        //----------------------------------------------------------------------------------------------------------------------
        //-----------------------------------------ESTIMATE TRANSFORM-----------------------------------------------------------

        _TransformationMatrix = this->estimate_transform(_filtered_correspondences);

        auto process_clock_esti = std::chrono::high_resolution_clock::now();
        process_duration = process_clock_esti - process_clock_rej;
        RCLCPP_INFO(this->_node->get_logger(), "Success : Transformation estimation");
        RCLCPP_INFO(this->_node->get_logger(), "Took : %.3f ms", process_duration.count());

        //----------------------------------------------------------------------------------------------------------------------
        //----------------------------------------------------------------------------------------------------------------------

        RCLCPP_INFO(this->_node->get_logger(), "Success : Estimating transform");
        RCLCPP_INFO(this->_node->get_logger(), "Original transformation (goal) : %s", Utils::matrix_to_string(_modif_transform).c_str());
        RCLCPP_INFO(this->_node->get_logger(), "Transformation Matrix found : %s", Utils::matrix_to_string(_TransformationMatrix).c_str());

        
        pcl::transformPointCloudWithNormals(*_input_cloud, *_transformed_cloud, _TransformationMatrix);
        
        TransformationError T_Error = Utils::transformation_matrix_eval(_modif_transform, _TransformationMatrix);
        translation_eval += T_Error.translation_error;
        rotation_eval += T_Error.rotation_error;
        RCLCPP_INFO(this->_node->get_logger(), "Translation error : %f", T_Error.translation_error);
        RCLCPP_INFO(this->_node->get_logger(), "Rotation error : %f", T_Error.rotation_error);

        RCLCPP_INFO(this->_node->get_logger(), "Publishing Results");
        this->publish_debug_pipeline();
        
        RCLCPP_INFO(this->_node->get_logger(), "Cleaning Loop");
        this->loop_cleaning();
        RCLCPP_INFO(this->_node->get_logger(), "Loop Cleaned");
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
    RCLCPP_INFO(this->_node->get_logger(), "Timer ended in : %.3f ms", duration_ms.count());
    RCLCPP_INFO(this->_node->get_logger(), "Time per iteration : %.3f ms", duration_ms.count()/_number_of_iterations);
    translation_eval = translation_eval/_number_of_iterations;
    RCLCPP_INFO(this->_node->get_logger(), "Final translation error : %f", translation_eval);
    rotation_eval = rotation_eval/_number_of_iterations;
    RCLCPP_INFO(this->_node->get_logger(), "Final rotation error : %f", rotation_eval);
    
    RCLCPP_INFO(this->_node->get_logger(), "Cleaning Pipeline");
    this->reset_object();
    RCLCPP_INFO(this->_node->get_logger(), "Pipeline Cleaned");
}

void loop_closure::reset_object(){
    this->loop_cleaning();

    if (_src_input_cloud) _src_input_cloud->clear();
    if (_src_reference_cloud) _src_reference_cloud->clear();
    
    _files_ok = true;
    _input_centroid = Eigen::Vector4f::Zero();
    _reference_centroid = Eigen::Vector4f::Zero();
}
void loop_closure::loop_cleaning(){
    if (_input_cloud) _input_cloud->clear();
    if (_reference_cloud) _reference_cloud->clear();
    if (_input_keypoints) _input_keypoints->clear();
    if (_reference_keypoints) _reference_keypoints->clear();
    if (_transformed_cloud) _transformed_cloud->clear();

    if (_correspondences) _correspondences->clear();
    if (_filtered_correspondences) _filtered_correspondences->clear();

    _modif_transform = Eigen::Matrix4f::Identity();
    _TransformationMatrix = Eigen::Matrix4f::Identity();
}



/*--------------------------------------------------------------------------------------------------------------------------------
---------------                                                                                         --------------------------
---------------                                                                                         --------------------------
---------------                                     PUBLISHER                                           --------------------------
---------------                                                                                         --------------------------
---------------                                                                                         --------------------------
--------------------------------------------------------------------------------------------------------------------------------*/

void loop_closure::publish_debug_pipeline() {
    if (!_node) return;

    double keypoints_size = 0.03;

    auto header_time = _node->now();
    std::string frame_id = "sonar3d";

    // 1. Publication des Nuages de Points (PointClouds)
    if (_input_cloud) {
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*_input_cloud, msg);
        msg.header.stamp = header_time;
        msg.header.frame_id = frame_id;
        _node->input_cloud_pub->publish(msg);
    }

    if (_reference_cloud) {
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*_reference_cloud, msg);
        msg.header.stamp = header_time;
        msg.header.frame_id = frame_id;
        _node->reference_cloud_pub->publish(msg);
    }

    if (_transformed_cloud) {
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*_transformed_cloud, msg);
        msg.header.stamp = header_time;
        msg.header.frame_id = frame_id;
        _node->transformed_cloud_pub->publish(msg);
    }



    // 2. Publication des Keypoints (via MarkerArray)
    auto create_keypoint_markers = [&](auto cloud, std::string ns, float r, float g, float b) {
        visualization_msgs::msg::MarkerArray ma;
        if (!cloud || cloud->empty()) return ma;

        visualization_msgs::msg::Marker m;
        m.header.frame_id = frame_id;
        m.header.stamp = header_time;
        m.ns = ns;
        m.id = 0;
        m.type = visualization_msgs::msg::Marker::POINTS;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = keypoints_size; m.scale.y = keypoints_size; 
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0;

        for (const auto& pt : cloud->points) {
            geometry_msgs::msg::Point p;
            p.x = pt.x; p.y = pt.y; p.z = pt.z;
            m.points.push_back(p);
        }
        ma.markers.push_back(m);
        return ma;
    };

    _node->input_keypoints_pub->publish(create_keypoint_markers(_input_keypoints, "input_kp", 0.0, 0.0, 1.0)); // Bleu
    _node->reference_keypoints_pub->publish(create_keypoint_markers(_reference_keypoints, "ref_kp", 1.0, 1.0, 0.0)); // Jaune

    // 3. Publication des Correspondances (Matches)
    visualization_msgs::msg::MarkerArray match_ma;
    
    // On combine les correspondances brutes et filtrées dans un seul MarkerArray
    // Marker pour les correspondances BRUTES (Rouge transparent)
    if (_correspondences) {
        visualization_msgs::msg::Marker raw_lines;
        raw_lines.header.frame_id = frame_id;
        raw_lines.header.stamp = header_time;
        raw_lines.ns = "raw_matches";
        raw_lines.id = 0;
        raw_lines.type = visualization_msgs::msg::Marker::LINE_LIST;
        raw_lines.scale.x = 0.01; // Lignes fines
        raw_lines.color.r = 1.0; raw_lines.color.g = 0.0; raw_lines.color.b = 0.0; raw_lines.color.a = 0.3;

        for (const auto& c : *_correspondences) {
            geometry_msgs::msg::Point p1, p2;
            p1.x = _input_cloud->points[c.index_query].x;
            p1.y = _input_cloud->points[c.index_query].y;
            p1.z = _input_cloud->points[c.index_query].z;
            p2.x = _reference_cloud->points[c.index_match].x;
            p2.y = _reference_cloud->points[c.index_match].y;
            p2.z = _reference_cloud->points[c.index_match].z;
            raw_lines.points.push_back(p1);
            raw_lines.points.push_back(p2);
        }
        match_ma.markers.push_back(raw_lines);
    }

    // Marker pour les correspondances FILTRÉES (Vert épais)
    if (_filtered_correspondences) {
        visualization_msgs::msg::Marker filtered_lines;
        filtered_lines.header.frame_id = frame_id;
        filtered_lines.header.stamp = header_time;
        filtered_lines.ns = "filtered_matches";
        filtered_lines.id = 1;
        filtered_lines.type = visualization_msgs::msg::Marker::LINE_LIST;
        filtered_lines.scale.x = 0.015; // Lignes plus épaisses
        filtered_lines.color.r = 0.0; filtered_lines.color.g = 1.0; filtered_lines.color.b = 0.0; filtered_lines.color.a = 1.0;

        for (const auto& c : *_filtered_correspondences) {
            geometry_msgs::msg::Point p1, p2;
            p1.x = _input_cloud->points[c.index_query].x;
            p1.y = _input_cloud->points[c.index_query].y;
            p1.z = _input_cloud->points[c.index_query].z;
            p2.x = _reference_cloud->points[c.index_match].x;
            p2.y = _reference_cloud->points[c.index_match].y;
            p2.z = _reference_cloud->points[c.index_match].z;
            filtered_lines.points.push_back(p1);
            filtered_lines.points.push_back(p2);
        }
        match_ma.markers.push_back(filtered_lines);
    }

    if (!match_ma.markers.empty()) {
        _node->match_pub->publish(match_ma);
    }
}