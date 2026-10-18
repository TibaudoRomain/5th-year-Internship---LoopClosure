#include <roller/loopclosure/benchmark.hpp>

namespace roller::loopclosure {

    LoopClosureBenchmark::LoopClosureBenchmark(const std::string& saving_dir)
        : saving_directory(saving_dir)
    {}

    void LoopClosureBenchmark::set_saving_directory(const std::string& saving_dir) {
        saving_directory = saving_dir;
    }

    void LoopClosureBenchmark::set_gt_thresholds(float gt_dist_thresh_m, float max_rte_m, float max_rre_deg) {
        gt_distance_threshold_m = gt_dist_thresh_m;
        max_rte_threshold_m = max_rte_m;
        max_rre_threshold_deg = max_rre_deg;
    }

    void LoopClosureBenchmark::log_description_time(double duration_ms) {
        metrics.total_description_time_ms += duration_ms;
    }

    void LoopClosureBenchmark::log_candidate_search_time(double duration_ms) {
        metrics.total_candidate_search_time_ms += duration_ms;
    }

    float LoopClosureBenchmark::compute_rte(const Eigen::Matrix4f& T_est, const Eigen::Matrix4f& T_gt) {
        Eigen::Vector3f t_est = T_est.block<3, 1>(0, 3);
        Eigen::Vector3f t_gt = T_gt.block<3, 1>(0, 3);
        return (t_est - t_gt).norm();
    }

    float LoopClosureBenchmark::compute_rre(const Eigen::Matrix4f& T_est, const Eigen::Matrix4f& T_gt) {
        Eigen::Matrix3f R_est = T_est.block<3, 3>(0, 0);
        Eigen::Matrix3f R_gt = T_gt.block<3, 3>(0, 0);
        Eigen::Matrix3f R_diff = R_est.transpose() * R_gt;

        float trace_val = R_diff.trace();
        float cos_theta = std::min(1.0f, std::max(-1.0f, (trace_val - 1.0f) * 0.5f));
        return std::acos(cos_theta) * 180.0f / static_cast<float>(M_PI);
    }

    CandidatePairEvaluation LoopClosureBenchmark::evaluate_pair(
        size_t new_submap_id, size_t old_submap_id,
        const Eigen::Matrix4f& new_gt_pose, const Eigen::Matrix4f& old_gt_pose,
        double descriptor_dist, int yaw_shift, bool is_candidate_selected
    ) {
        CandidatePairEvaluation eval;
        eval.new_submap_id = new_submap_id;
        eval.old_submap_id = old_submap_id;
        eval.descriptor_dist = descriptor_dist;
        eval.yaw_shift = yaw_shift;
        eval.is_candidate_selected = is_candidate_selected;
        eval.registration_attempted = false;
        eval.registration_success = false;
        eval.n_correspondences = 0;
        eval.n_inliers = 0;
        eval.inlier_ratio = 0.0f;
        eval.estimated_transform = Eigen::Matrix4f::Identity();
        eval.rte_m = 0.0f;
        eval.rre_deg = 0.0f;
        eval.reg_accurate = false;
        eval.match_time_ms = 0.0;
        eval.estimate_time_ms = 0.0;

        // Compute Ground Truth relative transformation: T_ij_gt = T_i_gt^-1 * T_j_gt
        eval.gt_transform = new_gt_pose.inverse() * old_gt_pose;

        // Ground Truth spatial distance between submaps
        Eigen::Vector3f new_pos_gt = new_gt_pose.block<3, 1>(0, 3);
        Eigen::Vector3f old_pos_gt = old_gt_pose.block<3, 1>(0, 3);
        eval.gt_distance_m = (new_pos_gt - old_pos_gt).norm();

        eval.is_true_loop_closure = (eval.gt_distance_m <= gt_distance_threshold_m);

        // Update confusion matrix counters
        metrics.total_pairs_evaluated++;
        if (eval.is_candidate_selected && eval.is_true_loop_closure) {
            metrics.tp++;
        } else if (eval.is_candidate_selected && !eval.is_true_loop_closure) {
            metrics.fp++;
        } else if (!eval.is_candidate_selected && eval.is_true_loop_closure) {
            metrics.fn++;
        } else {
            metrics.tn++;
        }

        return eval;
    }

    void LoopClosureBenchmark::record_registration(
        CandidatePairEvaluation& eval,
        bool registration_success,
        int n_corr, int n_inliers,
        const Eigen::Matrix4f& estimated_transform,
        double match_time_ms, double estimate_time_ms
    ) {
        eval.registration_attempted = true;
        eval.registration_success = registration_success;
        eval.n_correspondences = n_corr;
        eval.n_inliers = n_inliers;
        eval.inlier_ratio = (n_corr > 0) ? static_cast<float>(n_inliers) / static_cast<float>(n_corr) : 0.0f;
        eval.estimated_transform = estimated_transform;
        eval.match_time_ms = match_time_ms;
        eval.estimate_time_ms = estimate_time_ms;

        metrics.total_matching_time_ms += match_time_ms;
        metrics.total_registration_time_ms += estimate_time_ms;
        metrics.total_registrations_attempted++;

        if (registration_success) {
            eval.rte_m = compute_rte(estimated_transform, eval.gt_transform);
            eval.rre_deg = compute_rre(estimated_transform, eval.gt_transform);

            eval.reg_accurate = (eval.rte_m <= max_rte_threshold_m && eval.rre_deg <= max_rre_threshold_deg);
            if (eval.reg_accurate) {
                metrics.total_registrations_accurate++;
            }
        }

        pair_evaluations.push_back(eval);
        update_summary();
    }

    void LoopClosureBenchmark::update_summary() {
        int denom_p = metrics.tp + metrics.fp;
        metrics.precision = (denom_p > 0) ? static_cast<float>(metrics.tp) / static_cast<float>(denom_p) : 0.0f;

        int denom_r = metrics.tp + metrics.fn;
        metrics.recall = (denom_r > 0) ? static_cast<float>(metrics.tp) / static_cast<float>(denom_r) : 0.0f;

        float pr_sum = metrics.precision + metrics.recall;
        metrics.f1_score = (pr_sum > 0.0f) ? 2.0f * (metrics.precision * metrics.recall) / pr_sum : 0.0f;

        metrics.registration_success_rate = (metrics.total_registrations_attempted > 0)
            ? static_cast<float>(metrics.total_registrations_accurate) / static_cast<float>(metrics.total_registrations_attempted)
            : 0.0f;

        float sum_rte = 0.0f;
        float sum_rre = 0.0f;
        float sum_inlier_ratio = 0.0f;
        int valid_reg_count = 0;

        for (const auto& eval : pair_evaluations) {
            if (eval.registration_attempted) {
                sum_inlier_ratio += eval.inlier_ratio;
                if (eval.registration_success) {
                    sum_rte += eval.rte_m;
                    sum_rre += eval.rre_deg;
                    valid_reg_count++;
                }
            }
        }

        metrics.mean_rte_m = (valid_reg_count > 0) ? sum_rte / valid_reg_count : 0.0f;
        metrics.mean_rre_deg = (valid_reg_count > 0) ? sum_rre / valid_reg_count : 0.0f;
        metrics.mean_inlier_ratio = (metrics.total_registrations_attempted > 0) ? sum_inlier_ratio / metrics.total_registrations_attempted : 0.0f;
    }

    void LoopClosureBenchmark::print_submap_report(size_t submap_id) const {
        std::cout << "\n================================================================================" << std::endl;
        std::cout << " [BENCHMARK REPORT] Submap " << submap_id << " Loop Closure Pipeline Evaluation" << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << "  - Pairs Evaluated         : " << metrics.total_pairs_evaluated << std::endl;
        std::cout << "  - True Positives (TP)     : " << metrics.tp << " (Correct Candidate & GT <= " << gt_distance_threshold_m << "m)" << std::endl;
        std::cout << "  - False Positives (FP)    : " << metrics.fp << " (Perceptual Alias / Wrong Candidate)" << std::endl;
        std::cout << "  - False Negatives (FN)    : " << metrics.fn << " (Missed Loop Closure)" << std::endl;
        std::cout << "  - Precision               : " << std::fixed << std::setprecision(2) << (metrics.precision * 100.0f) << "%" << std::endl;
        std::cout << "  - Recall                  : " << std::fixed << std::setprecision(2) << (metrics.recall * 100.0f) << "%" << std::endl;
        std::cout << "  - F1 Score                : " << std::fixed << std::setprecision(4) << metrics.f1_score << std::endl;
        std::cout << "  - Registration Accuracy   : " << (metrics.registration_success_rate * 100.0f) << "% (RTE <= " << max_rte_threshold_m << "m, RRE <= " << max_rre_threshold_deg << "deg)" << std::endl;
        std::cout << "  - Mean Relative Pos Error : RTE = " << metrics.mean_rte_m << " m, RRE = " << metrics.mean_rre_deg << " deg" << std::endl;
        std::cout << "================================================================================\n" << std::endl;
    }

    void LoopClosureBenchmark::print_final_summary() const {
        print_submap_report(metrics.total_submaps);
    }

    void LoopClosureBenchmark::export_csv() {
        if (saving_directory.empty()) return;

        std::string benchmark_dir = saving_directory + "/benchmark/";
        std::filesystem::create_directories(benchmark_dir);

        // Export Pairs Evaluation CSV
        std::ofstream pairs_file(benchmark_dir + "loopclosure_pairs.csv", std::ios::trunc);
        if (pairs_file.is_open()) {
            pairs_file << "new_submap_id,old_submap_id,gt_distance_m,is_true_loop,descriptor_dist,yaw_shift,candidate_selected,reg_attempted,reg_success,n_corr,n_inliers,inlier_ratio,rte_m,rre_deg,reg_accurate,match_ms,estimate_ms\n";
            for (const auto& eval : pair_evaluations) {
                pairs_file << eval.new_submap_id << ","
                           << eval.old_submap_id << ","
                           << eval.gt_distance_m << ","
                           << (eval.is_true_loop_closure ? 1 : 0) << ","
                           << eval.descriptor_dist << ","
                           << eval.yaw_shift << ","
                           << (eval.is_candidate_selected ? 1 : 0) << ","
                           << (eval.registration_attempted ? 1 : 0) << ","
                           << (eval.registration_success ? 1 : 0) << ","
                           << eval.n_correspondences << ","
                           << eval.n_inliers << ","
                           << eval.inlier_ratio << ","
                           << eval.rte_m << ","
                           << eval.rre_deg << ","
                           << (eval.reg_accurate ? 1 : 0) << ","
                           << eval.match_time_ms << ","
                           << eval.estimate_time_ms << "\n";
            }
            pairs_file.close();
        }

        // Export Summary CSV
        std::ofstream summary_file(benchmark_dir + "loopclosure_summary.csv", std::ios::trunc);
        if (summary_file.is_open()) {
            summary_file << "total_submaps,total_pairs,tp,fp,fn,tn,precision,recall,f1_score,total_reg_attempted,total_reg_accurate,reg_success_rate,mean_rte_m,mean_rre_deg,mean_inlier_ratio,desc_time_ms,candidate_search_time_ms,matching_time_ms,registration_time_ms\n";
            summary_file << metrics.total_submaps << ","
                         << metrics.total_pairs_evaluated << ","
                         << metrics.tp << ","
                         << metrics.fp << ","
                         << metrics.fn << ","
                         << metrics.tn << ","
                         << metrics.precision << ","
                         << metrics.recall << ","
                         << metrics.f1_score << ","
                         << metrics.total_registrations_attempted << ","
                         << metrics.total_registrations_accurate << ","
                         << metrics.registration_success_rate << ","
                         << metrics.mean_rte_m << ","
                         << metrics.mean_rre_deg << ","
                         << metrics.mean_inlier_ratio << ","
                         << metrics.total_description_time_ms << ","
                         << metrics.total_candidate_search_time_ms << ","
                         << metrics.total_matching_time_ms << ","
                         << metrics.total_registration_time_ms << "\n";
            summary_file.close();
        }

        generate_python_plotter();
    }

    void LoopClosureBenchmark::generate_python_plotter() const {
        if (saving_directory.empty()) return;
        std::string benchmark_dir = saving_directory + "/benchmark/";
        std::string script_path = benchmark_dir + "plot_benchmark.py";

        std::ofstream py(script_path, std::ios::trunc);
        if (!py.is_open()) return;

        py << R"(#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

pairs_path = 'loopclosure_pairs.csv'
summary_path = 'loopclosure_summary.csv'

if os.path.exists(pairs_path):
    df = pd.read_csv(pairs_path)
    print(f"Loaded {len(df)} candidate evaluation pairs.")
    
    plt.figure(figsize=(12, 5))
    
    # 1. Descriptor Distance vs Ground Truth Distance
    plt.subplot(1, 2, 1)
    tp_df = df[df['is_true_loop'] == 1]
    fp_df = df[df['is_true_loop'] == 0]
    
    plt.scatter(tp_df['gt_distance_m'], tp_df['descriptor_dist'], color='green', alpha=0.7, label='True Loop Pair')
    plt.scatter(fp_df['gt_distance_m'], fp_df['descriptor_dist'], color='red', alpha=0.5, label='Non-Loop Pair')
    plt.xlabel('Ground Truth Distance (m)')
    plt.ylabel('Descriptor Distance (NDTMC)')
    plt.title('Descriptor Distance vs GT Distance')
    plt.legend()
    plt.grid(True)
    
    # 2. Registration Error Distribution (RTE vs RRE)
    plt.subplot(1, 2, 2)
    reg_df = df[df['reg_attempted'] == 1]
    if not reg_df.empty:
        plt.scatter(reg_df['rte_m'], reg_df['rre_deg'], c=reg_df['inlier_ratio'], cmap='viridis', s=40)
        plt.colorbar(label='Inlier Ratio')
        plt.xlabel('Relative Translation Error (m)')
        plt.ylabel('Relative Rotation Error (deg)')
        plt.title('Registration Accuracy (RTE vs RRE)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('loopclosure_benchmark_analysis.png', dpi=300)
    print("Saved plot to loopclosure_benchmark_analysis.png")
)";
        py.close();
    }

} // namespace roller::loopclosure
