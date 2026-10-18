#include "roller/loopclosure/cuda/normals.cuh"
#include "roller/loopclosure/cuda/kernels.cuh"
#include "roller/loopclosure/cuda/pointcloud.cuh"

#include <iostream>

namespace roller::loopclosure::cuda {

    void GaussianFilter(nanovdb::FloatGrid* gpu_Grid_Ptr, nanovdb::CoordBBox bbox, int width, int repetitions){
        int dimX = bbox.max().x() - bbox.min().x() + 1;
        int dimY = bbox.max().y() - bbox.min().y() + 1;
        int dimZ = bbox.max().z() - bbox.min().z() + 1;

        nanovdb::Coord bbox_origin = bbox.min();
        float* d_temp_buffer;
        cudaMalloc(&d_temp_buffer, dimX * dimY * dimZ * sizeof(float));

        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((dimX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (dimY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (dimZ + threadsPerBlock.z - 1) / threadsPerBlock.z);

        for (int i = 0; i < repetitions; ++i) {
            GaussianFilterKernel<0><<<numBlocks, threadsPerBlock>>>(gpu_Grid_Ptr, d_temp_buffer, dimX, dimY, dimZ, bbox_origin, width);
            cudaDeviceSynchronize();
            reinjectToNano<<<numBlocks, threadsPerBlock>>>(d_temp_buffer, gpu_Grid_Ptr, dimX, dimY, dimZ, bbox_origin);
            cudaDeviceSynchronize();

            GaussianFilterKernel<1><<<numBlocks, threadsPerBlock>>>(gpu_Grid_Ptr, d_temp_buffer, dimX, dimY, dimZ, bbox_origin, width);
            cudaDeviceSynchronize();
            reinjectToNano<<<numBlocks, threadsPerBlock>>>(d_temp_buffer, gpu_Grid_Ptr, dimX, dimY, dimZ, bbox_origin);
            cudaDeviceSynchronize();

            GaussianFilterKernel<2><<<numBlocks, threadsPerBlock>>>(gpu_Grid_Ptr, d_temp_buffer, dimX, dimY, dimZ, bbox_origin, width);
            cudaDeviceSynchronize();
            reinjectToNano<<<numBlocks, threadsPerBlock>>>(d_temp_buffer, gpu_Grid_Ptr, dimX, dimY, dimZ, bbox_origin);
            cudaDeviceSynchronize();
        }

        cudaFree(d_temp_buffer);
    }

    void NormalAcquisition(
        PointNormal_GPU* dest, nanovdb::FloatGrid* Grid, nanovdb::CoordBBox bbox,
        float dist_to_sdf, float VoxelSize, int* normalCounter)
    {        
        if (!Grid) {
            std::cerr << "[NormalAcquisition] : Grid is null!" << std::endl;
            return;
        }

        nanovdb::Coord bbox_min = bbox.min();
        nanovdb::Coord bbox_max = bbox.max();

        int dimX = bbox_max.x() - bbox_min.x() + 1;
        int dimY = bbox_max.y() - bbox_min.y() + 1;
        int dimZ = bbox_max.z() - bbox_min.z() + 1;

        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((dimX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (dimY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (dimZ + threadsPerBlock.z - 1) / threadsPerBlock.z);

        float max_dist = (dist_to_sdf > 0.0f) ? dist_to_sdf : 0.3f;
        std::cout << "[NormalAcquisition] : Starting Kernel!" << std::endl;
        extractTSDFSurfaceKernel<<<numBlocks, threadsPerBlock>>>(Grid, dest, normalCounter, max_dist, VoxelSize, bbox_min, dimX, dimY, dimZ);

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            std::cerr << "[NormalAcquisition] : Kernel launch error: " 
                    << cudaGetErrorString(launch_err) << std::endl;
            return;
        }

        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            std::cerr << "[NormalAcquisition] : Kernel execution error: " 
                    << cudaGetErrorString(sync_err) << std::endl;
        }
        std::cout << "[NormalAcquisition] : Finished!" << std::endl;
    }

    void CurvAcquisition(
        PointNormal_GPU* inp_dest, const nanovdb::UInt32Grid* normalMap, int normalCount, float voxelSize)
    {        
        int threads = 512;
        int blocks = (normalCount + threads - 1) / threads;
        
        curvCalcsKernel<<<blocks, threads>>>(inp_dest, normalMap, normalCount, voxelSize);
        cudaDeviceSynchronize();
    }

} // namespace roller::loopclosure::cuda
