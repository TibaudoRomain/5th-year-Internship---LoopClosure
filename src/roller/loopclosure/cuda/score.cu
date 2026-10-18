#include "roller/loopclosure/cuda/score.cuh"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <cusolverDn.h>

namespace roller::loopclosure::cuda {

    __global__ void computePatchCovariancesKernel(
        const float3* d_surface,
        int surface_count,
        const Patch* d_patches,
        const int* d_patch_indices,
        int query_count,
        PatchCovariance* d_covariances_out
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= query_count) return;

        Patch patch = d_patches[idx];
        PatchCovariance cov{};
        cov.point_count = patch.count;

        if (patch.count < 3 || patch.offset < 0 || !d_patch_indices) {
            cov.centroid = patch.query_pt;
            d_covariances_out[idx] = cov;
            return;
        }

        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        int offset = patch.offset;
        for (int i = 0; i < patch.count; ++i) {
            int surf_idx = d_patch_indices[offset + i];
            if (surf_idx >= 0 && surf_idx < surface_count) {
                float3 pt = d_surface[surf_idx];
                sum_x += pt.x;
                sum_y += pt.y;
                sum_z += pt.z;
            }
        }

        double inv_n = 1.0 / static_cast<double>(patch.count);
        float3 centroid = make_float3(
            static_cast<float>(sum_x * inv_n),
            static_cast<float>(sum_y * inv_n),
            static_cast<float>(sum_z * inv_n)
        );
        cov.centroid = centroid;

        double cxx = 0.0, cxy = 0.0, cxz = 0.0;
        double cyy = 0.0, cyz = 0.0, czz = 0.0;

        for (int i = 0; i < patch.count; ++i) {
            int surf_idx = d_patch_indices[offset + i];
            if (surf_idx >= 0 && surf_idx < surface_count) {
                float3 pt = d_surface[surf_idx];
                double dx = pt.x - centroid.x;
                double dy = pt.y - centroid.y;
                double dz = pt.z - centroid.z;

                cxx += dx * dx;
                cxy += dx * dy;
                cxz += dx * dz;
                cyy += dy * dy;
                cyz += dy * dz;
                czz += dz * dz;
            }
        }

        cov.cxx = static_cast<float>(cxx * inv_n);
        cov.cxy = static_cast<float>(cxy * inv_n);
        cov.cxz = static_cast<float>(cxz * inv_n);
        cov.cyy = static_cast<float>(cyy * inv_n);
        cov.cyz = static_cast<float>(cyz * inv_n);
        cov.czz = static_cast<float>(czz * inv_n);

        d_covariances_out[idx] = cov;
    }

    void computePatchCovariancesGPU(
        const float3* d_surface,
        int surface_count,
        const Patch* d_patches,
        const int* d_patch_indices,
        int query_count,
        PatchCovariance* d_covariances_out,
        cudaStream_t stream
    ) {
        if (query_count <= 0 || !d_patches || !d_covariances_out) return;

        int threads = 256;
        int blocks = (query_count + threads - 1) / threads;

        computePatchCovariancesKernel<<<blocks, threads, 0, stream>>>(
            d_surface, surface_count, d_patches, d_patch_indices, query_count, d_covariances_out
        );
    }

    __device__ inline void computeEigenvalues3x3Symmetric(
        float cxx, float cxy, float cxz,
        float cyy, float cyz, float czz,
        float &e1, float &e2, float &e3
    ) {
        float tr = cxx + cyy + czz;
        float m11 = cyy * czz - cyz * cyz;
        float m22 = cxx * czz - cxz * cxz;
        float m33 = cxx * cyy - cxy * cxy;
        float m12 = cxz * cyz - cxy * czz;
        float m13 = cxy * cyz - cxz * cyy;
        float det = cxx * m11 + cxy * m12 + cxz * m13;
        float m2 = m11 + m22 + m33;

        float p = tr * tr - 3.0f * m2;
        float q = 2.0f * tr * tr * tr - 9.0f * tr * m2 + 27.0f * det;

        if (p < 1e-10f) {
            e1 = fmaxf(tr / 3.0f, 0.0f);
            e2 = e1;
            e3 = e1;
            return;
        }

        float phi = acosf(fminf(fmaxf(q / (2.0f * sqrtf(p * p * p)), -1.0f), 1.0f)) / 3.0f;
        float sqrt_p = sqrtf(p);

        float l1 = (tr + 2.0f * sqrt_p * cosf(phi)) / 3.0f;
        float l2 = (tr + 2.0f * sqrt_p * cosf(phi + 2.0943951023931953f)) / 3.0f;
        float l3 = (tr + 2.0f * sqrt_p * cosf(phi + 4.1887902047863905f)) / 3.0f;

        e1 = fmaxf(l1, fmaxf(l2, l3));
        e3 = fminf(l1, fminf(l2, l3));
        e2 = tr - e1 - e3;
    }

    __global__ void computePatchScoresKernel(
        const PatchCovariance* d_covariances,
        int query_count,
        float* d_scores_out
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= query_count) return;

        PatchCovariance cov = d_covariances[idx];
        if (cov.point_count < 3) {
            d_scores_out[idx] = 0.0f;
            return;
        }

        float e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
        computeEigenvalues3x3Symmetric(cov.cxx, cov.cxy, cov.cxz, cov.cyy, cov.cyz, cov.czz, e1, e2, e3);

        float sum_e = e1 + e2 + e3;
        float score = 0.0f;
        if (sum_e > 1e-8f && e3 > 0.0f) {
            score = 3.0f * e3 / sum_e;
        }
        d_scores_out[idx] = score;
    }

    void computePatchScoresGPU(
        const PatchCovariance* d_covariances,
        int query_count,
        float* d_scores_out,
        cudaStream_t stream
    ) {
        if (query_count <= 0 || !d_covariances || !d_scores_out) return;

        int threads = 256;
        int blocks = (query_count + threads - 1) / threads;

        computePatchScoresKernel<<<blocks, threads, 0, stream>>>(
            d_covariances, query_count, d_scores_out
        );
    }

} // namespace roller::loopclosure::cuda
