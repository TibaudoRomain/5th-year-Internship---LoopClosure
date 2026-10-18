#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <filesystem>

namespace roller::loopclosure {

    struct CandidatePairEvaluation {
        size_t new_submap_id;
        size_t old_submap_id;
        
        // Ground truth relative transform & distance
        float gt_distance_m;
        bool is_true_loop_closure; // gt_distance_m <= gt_distance_threshold
        
        // Descriptor / Candidate Search Stage
        double descriptor_dist;
        int yaw_shift;
        bool is_candidate_selected;
        
        // Registration Stage
        bool registration_attempted;
        bool registration_success;
        int n_correspondences;
        int n_inliers;
        float inlier_ratio;
        
        Eigen::Matrix4f estimated_transform;
        Eigen::Matrix4f gt_transform;
        
        float rte_m;       // Relative Translation Error (meters)
        float rre_deg;     // Relative Rotation Error (degrees)
        bool reg_accurate; // rte <= rte_threshold && rre <= rre_threshold
        
        // Latency
        double match_time_ms;
        double estimate_time_ms;
    };

    struct LoopClosureMetrics {
        int total_submaps = 0;
        int total_pairs_evaluated = 0;
        
        // Candidate Search Confusion Matrix
        int tp = 0; // True Positive: candidate selected & GT distance <= thresh
        int fp = 0; // False Positive: candidate selected & GT distance > thresh (perceptual alias)
        int fn = 0; // False Negative: GT distance <= thresh & candidate NOT selected
        int tn = 0; // True Negative: GT distance > thresh & candidate NOT selected
        
        float precision = 0.0f;
        float recall = 0.0f;
        float f1_score = 0.0f;
        
        // Registration Quality
        int total_registrations_attempted = 0;
        int total_registrations_accurate = 0; // RTE <= 1m, RRE <= 5 deg
        float registration_success_rate = 0.0f;
        
        float mean_rte_m = 0.0f;
        float mean_rre_deg = 0.0f;
        float mean_inlier_ratio = 0.0f;
        
        // Pipeline Latencies (ms)
        double total_description_time_ms = 0.0;
        double total_candidate_search_time_ms = 0.0;
        double total_matching_time_ms = 0.0;
        double total_registration_time_ms = 0.0;
    };

    class LoopClosureBenchmark {
    public:
        LoopClosureBenchmark(const std::string& saving_dir = "");
        ~LoopClosureBenchmark() = default;

        void set_saving_directory(const std::string& saving_dir);
        void set_gt_thresholds(float gt_dist_thresh_m = 3.0f, float max_rte_m = 1.0f, float max_rre_deg = 5.0f);

        void log_description_time(double duration_ms);
        void log_candidate_search_time(double duration_ms);

        CandidatePairEvaluation evaluate_pair(
            size_t new_submap_id, size_t old_submap_id,
            const Eigen::Matrix4f& new_gt_pose, const Eigen::Matrix4f& old_gt_pose,
            double descriptor_dist, int yaw_shift, bool is_candidate_selected
        );

        void record_registration(
            CandidatePairEvaluation& eval,
            bool registration_success,
            int n_corr, int n_inliers,
            const Eigen::Matrix4f& estimated_transform,
            double match_time_ms, double estimate_time_ms
        );

        void update_summary();
        void print_submap_report(size_t submap_id) const;
        void print_final_summary() const;
        void export_csv();
        void generate_python_plotter() const;

        const LoopClosureMetrics& get_metrics() const { return metrics; }
        const std::vector<CandidatePairEvaluation>& get_pair_evaluations() const { return pair_evaluations; }

    private:
        std::string saving_directory;
        float gt_distance_threshold_m = 3.0f;
        float max_rte_threshold_m = 1.0f;
        float max_rre_threshold_deg = 5.0f;

        LoopClosureMetrics metrics;
        std::vector<CandidatePairEvaluation> pair_evaluations;

        static float compute_rte(const Eigen::Matrix4f& T_est, const Eigen::Matrix4f& T_gt);
        static float compute_rre(const Eigen::Matrix4f& T_est, const Eigen::Matrix4f& T_gt);
    };

} // namespace roller::loopclosure
