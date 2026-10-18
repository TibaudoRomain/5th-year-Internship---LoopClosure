#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/util/HostBuffer.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <roller/submap.hpp>
#include <roller/loopclosure/cuda/types.hpp>

namespace roller::loopclosure {
    class CudaCore;
    /// Host-visible wrapper around the CUDA loop-closure pipeline.
    /// Only the public CPU interface is exposed here.
    class CudaCore_Wrapper {
    public:
        CudaCore_Wrapper(float voxel_size);
        ~CudaCore_Wrapper();

        CudaResult LDPipeline_device(const Submap& submap, const std::vector<Keyframe>& submap_keyframes, const nanovdb::GridHandle<nanovdb::HostBuffer>& sdf);

    private:
        std::unique_ptr<CudaCore> impl_;
    };
}
