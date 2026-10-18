#pragma once

#include <roller/loopclosure/host_pipeline.hpp>
#include <unordered_map>
#include <flann/util/serialization.h>
namespace flann {
namespace serialization {
template <typename K, typename V>
struct Serializer<std::unordered_map<K, V>> {
    template <typename InputArchive>
    static void load(InputArchive& ar, std::unordered_map<K, V>& map) {
        size_t size;
        ar & size;
        map.clear();
        for (size_t i = 0; i < size; ++i) {
            K k; V v;
            ar & k; ar & v;
            map[k] = v;
        }
    }
    template <typename OutputArchive>
    static void save(OutputArchive& ar, const std::unordered_map<K, V>& map) {
        size_t size = map.size();
        ar & size;
        for (const auto& kv : map) {
            ar & const_cast<K&>(kv.first);
            ar & const_cast<V&>(kv.second);
        }
    }
};
}
}
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/point_representation.h>
#include <type_traits>

namespace roller::loopclosure{
    
    template <typename DescriptorT>
    void loopclosure_master::match_features_lr(
        typename pcl::PointCloud<DescriptorT>::Ptr input, 
        typename pcl::PointCloud<DescriptorT>::Ptr reference, 
        pcl::Correspondences& output, double lowe_ratio){
        output.clear();
        pcl::search::KdTree<DescriptorT> descriptor_kdtree;
        descriptor_kdtree.setInputCloud(reference);
        double sq_lowe_ratio = lowe_ratio * lowe_ratio;
        for (size_t i = 0; i < input->size(); ++i) {
            bool is_nan = false;
            if constexpr (std::is_same_v<DescriptorT, pcl::SHOT352>) {is_nan = std::isnan(input->points[i].descriptor[0]);} 
            else {is_nan = std::isnan(input->points[i].histogram[0]);}
            if (is_nan) continue;
            std::vector<int> nearest_indices(2);
            std::vector<float> nearest_sq_distances(2);
            int found = descriptor_kdtree.nearestKSearch(input->points[i], 2, nearest_indices, nearest_sq_distances);
            if (found == 2) {
                if (nearest_sq_distances[0] < sq_lowe_ratio * nearest_sq_distances[1]) {
                    pcl::Correspondence corr;
                    corr.index_query = i;                     
                    corr.index_match = nearest_indices[0];    
                    corr.distance = nearest_sq_distances[0];
                    output.push_back(corr);
                }
            } 
            else if (found == 1) {
                pcl::Correspondence corr;
                corr.index_query = i;
                corr.index_match = nearest_indices[0];
                corr.distance = nearest_sq_distances[0];
                output.push_back(corr);
            }
        }
        if (Config::match_features::reciprocal) {
            pcl::Correspondences reciprocal_output;
            pcl::search::KdTree<DescriptorT> reverse_kdtree;
            reverse_kdtree.setInputCloud(input);
            for (const auto& corr : output) {
                std::vector<int> rev_indices(1);
                std::vector<float> rev_sq_distances(1);
                
                if (reverse_kdtree.nearestKSearch(reference->points[corr.index_match], 1, rev_indices, rev_sq_distances) > 0) {
                    if (rev_indices[0] == corr.index_query) {
                        reciprocal_output.push_back(corr);
                    }
                }
            }
            output = reciprocal_output;
        }
    }

    template <typename DescriptorT>
    void loopclosure_master::match_features_pcl(
        typename pcl::PointCloud<DescriptorT>::Ptr input, 
        typename pcl::PointCloud<DescriptorT>::Ptr reference, 
        pcl::Correspondences& output){
        pcl::registration::CorrespondenceEstimation<DescriptorT, DescriptorT> est;
        est.setInputSource (input);
        est.setInputTarget (reference);
        if(Config::match_features::reciprocal){est.determineReciprocalCorrespondences(output);}
        else{est.determineCorrespondences(output);}
    }

    template <typename DescriptorT>
    void loopclosure_master::match_features_sinkhorn(
        typename pcl::PointCloud<DescriptorT>::Ptr input, 
        typename pcl::PointCloud<DescriptorT>::Ptr reference, 
        pcl::Correspondences& output,
        float epsilon,       
        float dustbin_score, 
        int max_iterations)
    {
        output.clear();

        int N = input ? static_cast<int>(input->size()) : 0;
        int M = reference ? static_cast<int>(reference->size()) : 0;

        std::cout << "[Sinkhorn Debug] Called with parameters:" << std::endl;
        std::cout << "  - epsilon        : " << epsilon << std::endl;
        std::cout << "  - dustbin_score  : " << dustbin_score << std::endl;
        std::cout << "  - max_iterations : " << max_iterations << std::endl;
        std::cout << "  - Input size (N) : " << N << std::endl;
        std::cout << "  - Ref size (M)   : " << M << std::endl;

        if (N == 0 || M == 0) {
            std::cout << "[Sinkhorn Debug] Aborting: empty cloud (N=" << N << ", M=" << M << ")" << std::endl;
            return;
        }


        // ---------------------------------------------------------
        // 1. Extraction Générique (Agnostique au type de descripteur)
        // ---------------------------------------------------------
        pcl::DefaultPointRepresentation<DescriptorT> pt_rep;
        int desc_size = pt_rep.getNumberOfDimensions();

        Eigen::MatrixXf src_mat(N, desc_size);
        for (int i = 0; i < N; ++i) {
            std::vector<float> feat(desc_size);
            pt_rep.copyToFloatArray(input->points[i], feat.data());
            for (int k = 0; k < desc_size; ++k) src_mat(i, k) = feat[k];
        }

        Eigen::MatrixXf tgt_mat(M, desc_size);
        for (int i = 0; i < M; ++i) {
            std::vector<float> feat(desc_size);
            pt_rep.copyToFloatArray(reference->points[i], feat.data());
            for (int k = 0; k < desc_size; ++k) tgt_mat(i, k) = feat[k];
        }

        // ---------------------------------------------------------
        // 2. Calcul vectorisé ultra-rapide de la Matrice de Coût L2
        // ---------------------------------------------------------
        Eigen::VectorXf src_sq = src_mat.rowwise().squaredNorm();
        Eigen::VectorXf tgt_sq = tgt_mat.rowwise().squaredNorm();
        
        // Astuce algébrique : (x - y)^2 = x^2 + y^2 - 2xy
        Eigen::MatrixXf C = -2.0f * (src_mat * tgt_mat.transpose());
        C.colwise() += src_sq;
        C.rowwise() += tgt_sq.transpose();
        
        // On fixe à 0 les micro-valeurs négatives dues à l'imprécision des flottants
        C = C.cwiseMax(0.0f); 

        // Normalisation de la matrice de coût
        float max_c = C.maxCoeff();
        if (max_c > 0) C /= max_c;

        // ---------------------------------------------------------
        // 3. Matrice d'Affinité S avec Dustbins (Poubelles)
        // ---------------------------------------------------------
        Eigen::MatrixXf S(N + 1, M + 1);
        S.block(0, 0, N, M) = -C; 
        
        S.col(M).setConstant(dustbin_score);
        S.row(N).setConstant(dustbin_score);
        S(N, M) = 0.0f; 

        // ---------------------------------------------------------
        // 4. Transformation en Probabilités et Sinkhorn-Knopp
        // ---------------------------------------------------------
        Eigen::MatrixXf P = ((S.array() - S.maxCoeff()) / epsilon).exp();

        Eigen::VectorXf a = Eigen::VectorXf::Ones(N + 1);
        a(N) = M;
        Eigen::VectorXf b = Eigen::VectorXf::Ones(M + 1);
        b(M) = N;

        Eigen::VectorXf u = Eigen::VectorXf::Ones(N + 1);
        Eigen::VectorXf v = Eigen::VectorXf::Ones(M + 1);

        for (int iter = 0; iter < max_iterations; ++iter) {
            u = a.array() / (P * v).array();
            v = b.array() / (P.transpose() * u).array();
        }

        Eigen::MatrixXf T = u.asDiagonal() * P * v.asDiagonal();

        // ---------------------------------------------------------
        // 5. Extraction du One-to-One
        // ---------------------------------------------------------
        for (int i = 0; i < N; ++i) {
            int best_j;
            float max_p_row = T.row(i).head(M).maxCoeff(&best_j); 

            if (max_p_row > T(i, M)) {
                int best_i;
                T.col(best_j).head(N).maxCoeff(&best_i);

                if (best_i == i && max_p_row > T(N, best_j)) {
                    pcl::Correspondence corr;
                    corr.index_query = i;
                    corr.index_match = best_j;
                    corr.distance = C(i, best_j); 
                    output.push_back(corr);
                }
            }
        }
    }


    template<typename DescriptorT>
    void loopclosure_master::init_globalDescriptor(references& reference){
        DescriptorT avrg_descriptor;

        size_t desc_size = DescriptorT::descriptorSize();
        auto cloud_ptr = std::get<typename pcl::PointCloud<DescriptorT>::Ptr>(reference.descriptors);
        size_t num_features = cloud_ptr->points.size();


        
        for (size_t i = 0; i < desc_size; i++) {
            if constexpr (std::is_same_v<DescriptorT, pcl::SHOT352>) {
                avrg_descriptor.descriptor[i] = 0.0f;
            } else {
                avrg_descriptor.histogram[i] = 0.0f;
            }
        }

        for (const DescriptorT& feature : cloud_ptr->points) {
            for (size_t i = 0; i < desc_size; i++) {
                if constexpr (std::is_same_v<DescriptorT, pcl::SHOT352>) {
                    avrg_descriptor.descriptor[i] += feature.descriptor[i];
                } else {
                    avrg_descriptor.histogram[i] += feature.histogram[i];
                }
            }
        }

        if (num_features > 0) {
            for (size_t i = 0; i < desc_size; i++) {
                if constexpr (std::is_same_v<DescriptorT, pcl::SHOT352>) {
                    avrg_descriptor.descriptor[i] /= num_features;
                } else {
                    avrg_descriptor.histogram[i] /= num_features;
                }
            }
        }
        
        reference.global_descriptor = avrg_descriptor;
    }
}

