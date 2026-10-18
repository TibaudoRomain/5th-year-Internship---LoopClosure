#include "roller/loopclosure/cuda/device_pipeline.cuh"

#include <math_constants.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/Ray.h>

namespace roller::loopclosure::cuda {
    // Kernel pour calculer les normales via le gradient NanoVDB
    __device__ float3 computeNormal(nanovdb::FloatGrid::AccessorType& acc, const nanovdb::Coord& ijk, float voxel_size, bool& is_valid) {
        float3 grad = make_float3(0.0f,0.0f,0.0f);

        is_valid = true;

        const float w[3] = {3.0f, 10.0f, 3.0f};
        const float d[3] = {-1.0f, 0.0f, 1.0f};

        float wx, dx, wy, dy, wz, dz;
        #pragma unroll
        for (int _z = -1; _z <= 1; ++_z){
            wz = w[_z + 1];
            dz = d[_z + 1];
            #pragma unroll
            for (int _y = -1; _y <= 1; ++_y){
                wy = w[_y + 1];
                dy = d[_y + 1];
                #pragma unroll
                for (int _x = -1; _x <= 1; ++_x){
                    wx = w[_x + 1];
                    dx = d[_x + 1];
                    nanovdb::Coord neighbor = ijk.offsetBy(_x, _y, _z);

                    if (!acc.isActive(neighbor)){
                        is_valid = false;
                        return make_float3(0.0f, 0.0f, 0.0f);
                    }
                    else{

                        float val = acc.getValue(neighbor);
                        if (fabsf(val) > voxel_size * 5.0f){                             
                            is_valid = false;
                            return make_float3(0.0f, 0.0f, 0.0f);
                        }
                        else{                             
                            grad.x += dx * wy * wz * val;
                            grad.y += wx * dy * wz * val;
                            grad.z += wx * wy * dz * val;
                        }
                    }
                }
            }
        }
        float lenSq = grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
        if (lenSq > 1e-8f) {
            float invLen = rsqrtf(lenSq); 
            return make_float3(grad.x * invLen, grad.y * invLen, grad.z * invLen);
        }
        is_valid = false;
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    __device__ int whichKeyframe(const int global_index, const int* submap_offsets, int nb_of_keyframe) {
        int kf = 0;
        while (kf < nb_of_keyframe && global_index >= submap_offsets[kf + 1]) {
            kf++;
        }
        return kf;
    }

    /// @brief Return the normalized vector
    __device__ float3 normalize(const int3 vector) {
        float lenSQ = vector.x*vector.x + vector.y*vector.y + vector.z*vector.z;
        if (lenSQ > 1e-8f){
            float invLen = rsqrtf(lenSQ); 
            return make_float3(vector.x * invLen, vector.y * invLen, vector.z * invLen);
        }
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    __device__ float3 normalize(const float3 vector) {
        float lenSQ = vector.x*vector.x + vector.y*vector.y + vector.z*vector.z;
        if (lenSQ > 1e-8f){
            float invLen = rsqrtf(lenSQ); 
            return make_float3(vector.x * invLen, vector.y * invLen, vector.z * invLen);
        }
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    template <bool squared>
    __device__ float len(const int3 vector) {
        float lenSQ = static_cast<int>(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);
        if constexpr(squared){return lenSQ;}
        else{return sqrtf(lenSQ);}
    }
    template __device__ float len<true>(const int3 vector);
    template __device__ float len<false>(const int3 vector);

    __device__ int3 to_index(const float3 point, float voxelSize) {
        return make_int3(   static_cast<int>(point.x / voxelSize),
                            static_cast<int>(point.y / voxelSize),
                            static_cast<int>(point.z / voxelSize));
    }
    __device__ __forceinline__ float dot(const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __device__ __forceinline__ float3 dif(const float3& a, const float3& b) {
        return make_float3(a.x - b.x , a.y - b.y , a.z - b.z);
    }

    __device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }




    //Param : range_around_point is in meters
    //Param : direction shouldn't be normalized as the distance is needed
    __device__ nanovdb::math::Ray<float> getRay(const int3 &start, const int3 &direction,  float voxelSize, 
        float range_around_point, float sonar_min_range, float sonar_max_range
    ){
        float search_window = range_around_point / voxelSize;
        float3 normalized_direction = normalize(direction);
        float distance_point_sensor = len<false>(direction);
        nanovdb::Vec3f start_ray(start.x, start.y, start.z);
        nanovdb::Vec3f direction_ray(normalized_direction.x, normalized_direction.y, normalized_direction.z);
        float born_inf = fmaxf(sonar_min_range/voxelSize, distance_point_sensor-search_window);
        float born_sup = fminf(sonar_max_range/voxelSize, distance_point_sensor+search_window);
        return nanovdb::math::Ray<float>(start_ray, direction_ray, born_inf, born_sup);
    }

    //Param : range_around_point is in meters
    //Param : normalized_direction should be normalized
    //Param : distance_point_sensor should be in index length
    __device__ nanovdb::math::Ray<float> getRay(const int3 &start, const float3 &normalized_direction, float distance_point_sensor, float voxelSize,
        float range_around_point = 0.2f, float sonar_min_range = 0.2f, float sonar_max_range = 15.0f
    ){
        float search_window = range_around_point / voxelSize;
        nanovdb::Vec3f start_ray(start.x, start.y, start.z);
        nanovdb::Vec3f direction_ray(normalized_direction.x, normalized_direction.y, normalized_direction.z);
        float born_inf = fmaxf(sonar_min_range/voxelSize, distance_point_sensor-search_window);
        float born_sup = fminf(sonar_max_range/voxelSize, distance_point_sensor+search_window);
        return nanovdb::math::Ray<float>(start_ray, direction_ray, born_inf, born_sup);
    }


    __device__ void computeDarbouxAngles(
    float3 p_source, float3 n_source, 
    float3 p_neighbor, float3 n_neighbor, 
    float& alpha, float& phi, float& theta
    ) {
        float3 diff = make_float3(p_neighbor.x - p_source.x, 
                                p_neighbor.y - p_source.y, 
                                p_neighbor.z - p_source.z);
        float dist = sqrtf(dot(diff, diff));

        if (dist < 1e-5f) {
            alpha = 0.0f; phi = 0.0f; theta = 0.0f;
            return;
        }

        float3 diff_norm = make_float3(diff.x / dist, diff.y / dist, diff.z / dist);

        float n1_dot_d = dot(n_source, diff_norm);
        float n2_dot_d = dot(n_neighbor, diff_norm);

        float angle1 = acosf(fminf(fmaxf(n1_dot_d, -1.0f), 1.0f));
        float angle2 = acosf(fminf(fmaxf(n2_dot_d, -1.0f), 1.0f));

        float3 n_i = n_source, n_j = n_neighbor;

        if (angle1 > angle2) {
            n_i = n_neighbor;  n_j = n_source;
            
            diff = make_float3(-diff.x, -diff.y, -diff.z);
            diff_norm = make_float3(-diff_norm.x, -diff_norm.y, -diff_norm.z);
        }

        float3 u = n_i;
        
        float3 v = normalize(cross(diff, u));

        float3 w = cross(u, v);

        alpha = dot(v, n_j);
        phi = dot(u, diff_norm);
        theta = atan2f(dot(w, n_j), dot(u, n_j));
    }

    /**
     * @brief Helper device function to compute the eigenvalues of a 3x3 symmetric matrix.
     * Uses the analytical Cardano's formula.
     */
    __device__ void computeEigenvalues3x3(
        float m11, float m12, float m13, 
        float m22, float m23, float m33, 
        float &l1, float &l2, float &l3) 
    {
        float p1 = m12 * m12 + m13 * m13 + m23 * m23;
        if (p1 == 0.0f) {
            l1 = m11; l2 = m22; l3 = m33;
        } else {
            float q = (m11 + m22 + m33) / 3.0f;
            float p2 = (m11 - q)*(m11 - q) + (m22 - q)*(m22 - q) + (m33 - q)*(m33 - q) + 2.0f * p1;
            float p = sqrtf(p2 / 6.0f);
            
            float inv_p = 1.0f / p;
            float b11 = (m11 - q) * inv_p; float b22 = (m22 - q) * inv_p; float b33 = (m33 - q) * inv_p;
            float b12 = m12 * inv_p;       float b13 = m13 * inv_p;       float b23 = m23 * inv_p;
            
            float r = b11 * (b22 * b33 - b23 * b23)
                    - b12 * (b12 * b33 - b13 * b23)
                    + b13 * (b12 * b23 - b13 * b22);
            r /= 2.0f;
            
            float phi;
            if (r <= -1.0f) phi = CUDART_PI_F / 3.0f;
            else if (r >= 1.0f) phi = 0.0f;
            else phi = acosf(r) / 3.0f;
            
            l1 = q + 2.0f * p * cosf(phi);
            l3 = q + 2.0f * p * cosf(phi + 2.0f * CUDART_PI_F / 3.0f);
            l2 = 3.0f * q - l1 - l3; // trace = l1 + l2 + l3
        }
        
        // Ensure l1 >= l2 >= l3
        if (l1 < l2) { float tmp = l1; l1 = l2; l2 = tmp; }
        if (l2 < l3) { float tmp = l2; l2 = l3; l3 = tmp; }
        if (l1 < l2) { float tmp = l1; l1 = l2; l2 = tmp; }
    }



    
    
    
    

    template <int AXIS>
    __global__ void GaussianFilterKernel(const nanovdb::NanoGrid<float>* deviceGrid, float* dest, 
        int dimX, int dimY, int dimZ,
        nanovdb::Coord bbox_origin, int width) 
    {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
            blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            
            // activeVoxelCount() renvoie un uint64_t, on utilise donc %llu
            printf("[DEBUG Gaussian GPU] (Axe %d) Voxels actifs dans la grille : %llu\n", 
                   AXIS, (unsigned long long)deviceGrid->activeVoxelCount());
        }

        auto acc = deviceGrid->getAccessor();
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= dimX || y >= dimY || z >= dimZ) return;

        nanovdb::Coord xyz(x+bbox_origin.x(), y+bbox_origin.y(), z+bbox_origin.z());
        if (!acc.isActive(xyz)) return;

        float sum = 0.0f;
        nanovdb::Coord neighbor;
        for(int off = -width; off<=width; off++){
            neighbor = xyz;
            if constexpr(AXIS == AXIS_X){ neighbor.x() += off; }
            else if constexpr(AXIS == AXIS_Y){ neighbor.y() += off; }
            else if constexpr(AXIS == AXIS_Z){ neighbor.z() += off; }
            float val;
            if (acc.probeValue(neighbor, val)) {
                sum += val;
            }
        }
        auto ArrayCoord = x + y * dimX + z * dimX * dimY;
        dest[ArrayCoord] = sum / static_cast<float>(2*width + 1);
    }
    template __global__ void GaussianFilterKernel<AXIS_X>(const nanovdb::NanoGrid<float>* deviceGrid, float* dest, int dimX, int dimY, int dimZ, nanovdb::Coord bbox_origin, int width);
    template __global__ void GaussianFilterKernel<AXIS_Y>(const nanovdb::NanoGrid<float>* deviceGrid, float* dest, int dimX, int dimY, int dimZ, nanovdb::Coord bbox_origin, int width);
    template __global__ void GaussianFilterKernel<AXIS_Z>(const nanovdb::NanoGrid<float>* deviceGrid, float* dest, int dimX, int dimY, int dimZ, nanovdb::Coord bbox_origin, int width);



    __global__ void reinjectToNano(float* inputBuffer, nanovdb::NanoGrid<float>* deviceGrid,
        int dimX, int dimY, int dimZ,
         nanovdb::Coord bbox_origin)
    {
        //to read values
        auto acc = deviceGrid->getAccessor();

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= dimX || y >= dimY || z >= dimZ) return;

        nanovdb::Coord xyz(x+bbox_origin.x(), y+bbox_origin.y(), z+bbox_origin.z());
        int index =  x + y*dimX + z*dimX*dimY;

        if (acc.isActive(xyz)){
            auto upper_node = deviceGrid->tree().root().probeChild(xyz);
            if (!upper_node) return;
            auto lower_node = upper_node->probeChild(xyz);
            if (!lower_node) return;
            auto leaf = lower_node->probeChild(xyz);
            if (!leaf) return;
            int offset = nanovdb::NanoLeaf<float>::CoordToOffset(xyz);
            leaf->setValueOnly(offset, inputBuffer[index]);
        }
    }

    __global__ void extractPositionsKernel(const PointNormal_GPU* input, nanovdb::Vec3f* output, int count) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count) {
            output[idx] = nanovdb::Vec3f(input[idx].pos.x, input[idx].pos.y, input[idx].pos.z);
        }
    }


    __global__ void normalCalcsKernel(const float4* submap, const int* submap_offsets, int nb_of_keyframe, int num_points,
        PointNormal_GPU* dest,
        const nanovdb::NanoGrid<float>* sdf_grid,
        const float3* sensor_xyz,
        int* normalCounter,
        float dist_to_sdf,
        float voxelSize
    ){
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
            blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            
            printf("[DEBUG Normals] Started Normal calculations");
        }
        
        int point_global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (point_global_index >= num_points) return;


        int KeyframeId = whichKeyframe(point_global_index, submap_offsets, nb_of_keyframe);
        auto acc = sdf_grid->getAccessor(); 
        
        float3 point_xyz = make_float3(submap[point_global_index].x,submap[point_global_index].y,submap[point_global_index].z);

        float3 sensor_pos = sensor_xyz[KeyframeId];
        int3 sensor_xyz_index = to_index(sensor_pos,voxelSize);
        int3 point_xyz_index = to_index(point_xyz,voxelSize);

        int3 vct_point_to_sensor = make_int3(point_xyz_index.x - sensor_xyz_index.x, point_xyz_index.y - sensor_xyz_index.y, point_xyz_index.z - sensor_xyz_index.z);
        float3 dir_point_to_sensor = normalize(vct_point_to_sensor);
        float d_PointToSensor = len<false>(vct_point_to_sensor);

        nanovdb::math::Ray<float> ray = getRay(sensor_xyz_index, dir_point_to_sensor, d_PointToSensor, voxelSize, dist_to_sdf);
        nanovdb::Coord point_on_tsdf; float d_SensorToTsdf; float crossingPointValue;


        if(nanovdb::math::zeroCrossing(ray, acc, point_on_tsdf, crossingPointValue, d_SensorToTsdf)){
            bool is_valid = false;
            float3 computed_normal = computeNormal(acc, point_on_tsdf, voxelSize, is_valid);

            if(is_valid){
                float d_PointToTsdf = fabsf(d_SensorToTsdf - d_PointToSensor);
                PointNormal_GPU result;
                result.normal = computed_normal;
                result.pos = make_float3(point_on_tsdf.x()* voxelSize, point_on_tsdf.y()* voxelSize, point_on_tsdf.z()* voxelSize);
                result.normal_trust = fmaxf(0.0f, 0.1f - d_PointToTsdf*voxelSize)/0.1f;
                
                int table_index = atomicAdd(normalCounter, 1);
                dest[table_index] = result;
            }
            
        }
    }


    __global__ void curvCalcsKernel(PointNormal_GPU* input_normals, const nanovdb::UInt32Grid* normalsSpatialMap, 
        int total_surface_points, float voxelSize
    ){
        int TPoint_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (TPoint_idx >= total_surface_points) return;

        float3 TPoint_pos = input_normals[TPoint_idx].pos;
        nanovdb::Coord TPoint_ijk(
            __float2int_rn(TPoint_pos.x), 
            __float2int_rn(TPoint_pos.y), 
            __float2int_rn(TPoint_pos.z)
        );
        auto acc = normalsSpatialMap->getAccessor(); 

        auto get_neighbor_normal = [&](const nanovdb::Coord& neighbor_coord) -> float3 {
            if (acc.isActive(neighbor_coord)) {
                uint32_t neighbor_dest_index = acc.getValue(neighbor_coord);
                return input_normals[neighbor_dest_index].normal;
            }
            return make_float3(0.0f, 0.0f, 0.0f); 
        };
        float3 Nx1 = get_neighbor_normal(TPoint_ijk.offsetBy( 1,  0,  0));
        float3 Nx2 = get_neighbor_normal(TPoint_ijk.offsetBy(-1,  0,  0));
        float3 Ny1 = get_neighbor_normal(TPoint_ijk.offsetBy( 0,  1,  0));
        float3 Ny2 = get_neighbor_normal(TPoint_ijk.offsetBy( 0, -1,  0));
        float3 Nz1 = get_neighbor_normal(TPoint_ijk.offsetBy( 0,  0,  1));
        float3 Nz2 = get_neighbor_normal(TPoint_ijk.offsetBy( 0,  0, -1));
        
        float dNx_dx = Nx1.x - Nx2.x;
        float dNy_dy = Ny1.y - Ny2.y;
        float dNz_dz = Nz1.z - Nz2.z;

        input_normals->curvature = fabsf(0.5f * (dNx_dx + dNy_dy + dNz_dz));
    }

    /**
     * @brief Kernel used to calculate harris scores for each points of the cloud
     * 
     * @param  grid: a nanovdb grid that has input's indexes has value
     * @param  input:  a buffer containing our points
     * @param  d_harrisScores:  a buffer aligned with the input which gives the scores, the output
     * @param  pointCount: the number of points in the point cloud
     * @param  searchRadius: a threshold used to filter out points with a score too high (statistical outliers)
     * @param  voxelSize: the size of a voxel
     * @param  k: trace factor
     * 
     */
    __global__ void Harris3DKernel(
        const nanovdb::UInt32Grid* grid,
        const PointNormal_GPU* input,
        float* d_harrisScores,
        float* d_scales,
        int pointCount,
        float searchRadius, float voxelSize,
        float k,
        bool scale_aware,
        int num_scales,
        float scale_multiplier)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= pointCount) return;

        PointNormal_GPU currentPoint = input[idx];
        float3 p = currentPoint.pos;
        
        // 5. Coalesced accessor & baseVoxel precalculation
        auto accessor = grid->getAccessor();
        nanovdb::Coord baseVoxel = nanovdb::Coord(floorf(p.x / voxelSize), floorf(p.y / voxelSize), floorf(p.z / voxelSize));

        int totalScales = scale_aware ? max(1, min(num_scales, 8)) : 1;
        float scales_eval[8];
        float scores_eval[8];

        float currentSearchRadius = searchRadius;
        float voxelSizeSq = voxelSize * voxelSize;

        for (int s = 0; s < totalScales; ++s) {
            scales_eval[s] = currentSearchRadius;
            scores_eval[s] = -1e9f;

            float m11 = 0.0f, m12 = 0.0f, m13 = 0.0f;
            float m22 = 0.0f, m23 = 0.0f;
            float m33 = 0.0f;

            int VoxelRadius = static_cast<int>(ceilf(currentSearchRadius / voxelSize));
            float m_radiusSq = currentSearchRadius * currentSearchRadius;
            int Num_Neighbor = 0;

            for (int dz = -VoxelRadius; dz <= VoxelRadius; ++dz) {
                for (int dy = -VoxelRadius; dy <= VoxelRadius; ++dy) {
                    for (int dx = -VoxelRadius; dx <= VoxelRadius; ++dx) {
                        int distVoxelSq = dx * dx + dy * dy + dz * dz;
                        if (static_cast<float>(distVoxelSq) * voxelSizeSq > m_radiusSq + 1e-4f) continue;

                        nanovdb::Coord offset = nanovdb::Coord(dx, dy, dz);
                        nanovdb::Coord currentNeighbor = baseVoxel + offset;

                        if (!accessor.isActive(currentNeighbor)) continue;

                        uint32_t neighborIdx = accessor.getValue(currentNeighbor);
                        if (neighborIdx >= pointCount) continue;

                        PointNormal_GPU neighbor = input[neighborIdx];
                        float3 np = neighbor.pos;

                        float dx_m = np.x - p.x;
                        float dy_m = np.y - p.y;
                        float dz_m = np.z - p.z;
                        float m_distSq = dx_m * dx_m + dy_m * dy_m + dz_m * dz_m;

                        if (m_distSq > 0.000001f && m_distSq <= m_radiusSq) {
                            float3 n = neighbor.normal;
                            m11 += n.x * n.x;  m12 += n.x * n.y;  m13 += n.x * n.z;
                            m22 += n.y * n.y;  m23 += n.y * n.z;
                            m33 += n.z * n.z;
                            Num_Neighbor++;
                        }
                    }
                }
            }

            // 4. Boundary & edge conditioning: require minimum neighbor count (at least 5 points)
            if (Num_Neighbor >= 3) {
                float inv_Num_Neighbor = 1.0f / (float)Num_Neighbor;
                m11 *= inv_Num_Neighbor;  m12 *= inv_Num_Neighbor;  m13 *= inv_Num_Neighbor;
                m22 *= inv_Num_Neighbor;  m23 *= inv_Num_Neighbor;
                m33 *= inv_Num_Neighbor;

                float det = m11 * (m22 * m33 - m23 * m23)
                          - m12 * (m12 * m33 - m13 * m23)
                          + m13 * (m12 * m23 - m13 * m22);
                float trace = m11 + m22 + m33;
                
                // 3. Density-aware normalized raw score
                scores_eval[s] = det - k * (trace * trace);
            }

            currentSearchRadius *= scale_multiplier;
        }

        // 1. Scale-space extremum detection: search for local peak across scale
        float bestScore = -1e9f;
        float bestScale = searchRadius;
        bool peak_found = false;

        for (int s = 0; s < totalScales; ++s) {
            float sc = scores_eval[s];
            if (sc <= -1e8f) continue;

            bool is_left_peak  = (s == 0 || sc >= scores_eval[s - 1]);
            bool is_right_peak = (s == totalScales - 1 || sc >= scores_eval[s + 1]);

            if (is_left_peak && is_right_peak) {
                if (sc > bestScore) {
                    bestScore = sc;
                    bestScale = scales_eval[s];
                    peak_found = true;
                }
            }
        }

        // Fallback to max score if no scale peak was detected
        if (!peak_found) {
            for (int s = 0; s < totalScales; ++s) {
                if (scores_eval[s] > bestScore || s == 0) {
                    bestScore = scores_eval[s];
                    bestScale = scales_eval[s];
                }
            }
        }

        d_harrisScores[idx] = bestScore;
        if (d_scales) d_scales[idx] = bestScale;
    }

    __global__ void ThresholdFilterKernel(
        const PointNormal_GPU* input,
        const float* d_scores,
        const float* d_scales,
        int Num_Keypoints_Max,
        int pointCount,
        float min_score_threshold,
        keypoints* d_outputKeypoints, 
        int* d_keypointCount)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= pointCount) return;

        float currentScore = d_scores[idx];

        if (currentScore > min_score_threshold)
        {
            int outputIdx = atomicAdd(d_keypointCount, 1);
            if(outputIdx < Num_Keypoints_Max){
                PointNormal_GPU pt = input[idx];
                keypoints kp;

                kp.pos = pt.pos;
                kp.normal = pt.normal;
                kp.normal_trust = pt.normal_trust;
                kp.score = currentScore;
                kp.scale = d_scales ? d_scales[idx] : 0.0f;
                kp.curvature = pt.curvature;
                kp.index_on_map = idx;

                d_outputKeypoints[outputIdx] = kp;
            }
        }
    }

    __global__ void NumberFilterKernel(
        const PointNormal_GPU* input,
        const float* d_scores,
        const float* d_scales,
        int Num_Keypoints,
        int pointCount,
        keypoints* d_outputKeypoints, 
        int* d_keypointCount)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= pointCount) return;

        float currentScore = d_scores[idx];

        if (currentScore > -1e8f)
        {
            int outputIdx = atomicAdd(d_keypointCount, 1);

            if(outputIdx < Num_Keypoints){
                PointNormal_GPU pt = input[idx];
                keypoints kp;

                kp.pos = pt.pos;
                kp.normal = pt.normal;
                kp.normal_trust = pt.normal_trust;
                kp.score = currentScore;
                kp.scale = d_scales ? d_scales[idx] : 0.0f;
                kp.curvature = pt.curvature;
                kp.index_on_map = idx;

                d_outputKeypoints[outputIdx] = kp;
            }
        }
    }

    __global__ void ISS3DKernel(
        const nanovdb::UInt32Grid* grid,
        const PointNormal_GPU* input,
        float* d_issScores,
        float* d_scales,
        int pointCount,
        float searchRadius, float voxelSize, 
        float gamma_21, float gamma_32,
        bool scale_aware, int num_scales, float scale_multiplier)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= pointCount) return;

        PointNormal_GPU currentPoint = input[idx];
        float3 p = currentPoint.pos;
        
        // 5. Coalesced accessor & baseVoxel precalculation
        auto accessor = grid->getAccessor();
        nanovdb::Coord baseVoxel = nanovdb::Coord(floorf(p.x / voxelSize), floorf(p.y / voxelSize), floorf(p.z / voxelSize));

        int totalScales = scale_aware ? max(1, min(num_scales, 8)) : 1;
        float scales_eval[8];
        float scores_eval[8];

        float currentSearchRadius = searchRadius;
        float voxelSizeSq = voxelSize * voxelSize;

        for (int s = 0; s < totalScales; ++s) {
            scales_eval[s] = currentSearchRadius;
            scores_eval[s] = -1e9f;

            float m11 = 0.0f, m12 = 0.0f, m13 = 0.0f;
            float m22 = 0.0f, m23 = 0.0f;
            float m33 = 0.0f;

            int VoxelRadius = static_cast<int>(ceilf(currentSearchRadius / voxelSize));
            float m_radiusSq = currentSearchRadius * currentSearchRadius;
            int Num_Neighbor = 0;

            for (int dz = -VoxelRadius; dz <= VoxelRadius; ++dz) {
                for (int dy = -VoxelRadius; dy <= VoxelRadius; ++dy) {
                    for (int dx = -VoxelRadius; dx <= VoxelRadius; ++dx) {
                        int distVoxelSq = dx * dx + dy * dy + dz * dz;
                        if (static_cast<float>(distVoxelSq) * voxelSizeSq > m_radiusSq + 1e-4f) continue;

                        nanovdb::Coord offset = nanovdb::Coord(dx, dy, dz);
                        nanovdb::Coord currentNeighbor = baseVoxel + offset;

                        if (!accessor.isActive(currentNeighbor)) continue;

                        uint32_t neighborIdx = accessor.getValue(currentNeighbor);
                        if (neighborIdx >= pointCount) continue;

                        PointNormal_GPU neighbor = input[neighborIdx];
                        float3 np = neighbor.pos;

                        float dx_m = np.x - p.x;
                        float dy_m = np.y - p.y;
                        float dz_m = np.z - p.z;
                        float m_distSq = dx_m * dx_m + dy_m * dy_m + dz_m * dz_m;

                        if (m_distSq > 0.000001f && m_distSq <= m_radiusSq) {
                            m11 += dx_m * dx_m;  m12 += dx_m * dy_m;  m13 += dx_m * dz_m;
                            m22 += dy_m * dy_m;  m23 += dy_m * dz_m;
                            m33 += dz_m * dz_m;
                            Num_Neighbor++;
                        }
                    }
                }
            }

            // 4. Boundary conditioning: require at least 5 neighbors
            if (Num_Neighbor >= 3) {
                float inv_Num_Neighbor = 1.0f / (float)Num_Neighbor; 
                m11 *= inv_Num_Neighbor;  m12 *= inv_Num_Neighbor;  m13 *= inv_Num_Neighbor;
                m22 *= inv_Num_Neighbor;  m23 *= inv_Num_Neighbor;
                m33 *= inv_Num_Neighbor;

                float l1, l2, l3;
                computeEigenvalues3x3(m11, m12, m13, m22, m23, m33, l1, l2, l3);

                if (l1 > 0.0f && l2 > 0.0f && (l2 / l1) <= gamma_21 && (l3 / l2) <= gamma_32) {
                    scores_eval[s] = l3; 
                }
            }

            currentSearchRadius *= scale_multiplier;
        }

        // 1. Scale-space extremum detection for ISS
        float bestScore = -1e9f;
        float bestScale = searchRadius;
        bool peak_found = false;

        for (int s = 0; s < totalScales; ++s) {
            float sc = scores_eval[s];
            if (sc <= -1e8f) continue;

            bool is_left_peak  = (s == 0 || sc >= scores_eval[s - 1]);
            bool is_right_peak = (s == totalScales - 1 || sc >= scores_eval[s + 1]);

            if (is_left_peak && is_right_peak) {
                if (sc > bestScore) {
                    bestScore = sc;
                    bestScale = scales_eval[s];
                    peak_found = true;
                }
            }
        }

        if (!peak_found) {
            for (int s = 0; s < totalScales; ++s) {
                if (scores_eval[s] > bestScore || s == 0) {
                    bestScore = scores_eval[s];
                    bestScale = scales_eval[s];
                }
            }
        }

        d_issScores[idx] = bestScore;
        if (d_scales) d_scales[idx] = bestScale;
    }

    //---------------------------------------------------------------------------------------------------------------------------
    //--------------------------BUILDING THE GRID--------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------------------------------
    __device__ uint64_t packCoordToKey(int x, int y, int z) {
        uint64_t kx = x & 0x1FFFFF;
        uint64_t ky = y & 0x1FFFFF;
        uint64_t kz = z & 0x1FFFFF;
        return (kx << 42) | (ky << 21) | kz;
    }

    //---------------------------------------------------------------------------------------------------------------------------
    // SCALE-AWARE 3D NON-MAXIMUM SUPPRESSION KERNEL
    //---------------------------------------------------------------------------------------------------------------------------
    __global__ void NMS3DKernel(
        const nanovdb::UInt32Grid* grid,
        const PointNormal_GPU* input,
        const float* d_scores,
        const float* d_scales,
        int pointCount,
        float nmsRadius,
        float voxelSize,
        float score_threshold,
        int Num_Keypoints_Max,
        keypoints* d_outputKeypoints,
        int* d_keypointCount
    ) {
        // Block-level shared memory reduction to eliminate VRAM atomic contention
        __shared__ int blockCount;
        __shared__ int blockBaseIdx;

        if (threadIdx.x == 0) {
            blockCount = 0;
            blockBaseIdx = 0;
        }
        __syncthreads();

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        bool is_valid = (idx < pointCount);
        float currentScore = 0.0f;
        float currentScale = nmsRadius;
        PointNormal_GPU selfPt;

        if (is_valid) {
            currentScore = d_scores[idx];
            if (d_scales) currentScale = d_scales[idx];

            if (currentScore < score_threshold || currentScore <= -1e8f) {
                is_valid = false;
            } else {
                selfPt = input[idx];
            }
        }

        bool is_local_max = false;

        // FAST EARLY EXIT: non-candidate points immediately skip NanoVDB accessor creation & 3D neighborhood loop
        if (is_valid) {
            float3 p = selfPt.pos;
            auto accessor = grid->getAccessor();
            
            float effectiveRadius = fmaxf(nmsRadius, currentScale * 0.75f);
            nanovdb::Coord currentVoxel = nanovdb::Coord(floorf(p.x / voxelSize), floorf(p.y / voxelSize), floorf(p.z / voxelSize));
            int VoxelRadius = static_cast<int>(ceilf(effectiveRadius / voxelSize));
            int maxVoxelRadiusSq = VoxelRadius * VoxelRadius;
            float effectiveRadiusSq = effectiveRadius * effectiveRadius;

            is_local_max = true;

            for (int dz = -VoxelRadius; dz <= VoxelRadius && is_local_max; ++dz) {
                for (int dy = -VoxelRadius; dy <= VoxelRadius && is_local_max; ++dy) {
                    for (int dx = -VoxelRadius; dx <= VoxelRadius && is_local_max; ++dx) {
                        int distVoxelSq = dx * dx + dy * dy + dz * dz;
                        if (distVoxelSq > maxVoxelRadiusSq) continue;

                        nanovdb::Coord offset = nanovdb::Coord(dx, dy, dz);
                        nanovdb::Coord currentNeighbor = currentVoxel + offset;

                        if (!accessor.isActive(currentNeighbor)) continue;

                        uint32_t neighborIdx = accessor.getValue(currentNeighbor);
                        if (neighborIdx >= pointCount || neighborIdx == idx) continue;

                        // Only compare points within comparable scale levels
                        if (d_scales) {
                            float neighborScale = d_scales[neighborIdx];
                            float scaleRatio = fmaxf(currentScale, neighborScale) / fmaxf(1e-4f, fminf(currentScale, neighborScale));
                            if (scaleRatio > 1.5f) continue;
                        }

                        PointNormal_GPU neighbor = input[neighborIdx];
                        float3 np = neighbor.pos;

                        float dx_m = np.x - p.x;
                        float dy_m = np.y - p.y;
                        float dz_m = np.z - p.z;
                        float distSq = dx_m * dx_m + dy_m * dy_m + dz_m * dz_m;

                        if (distSq <= effectiveRadiusSq) {
                            float neighborScore = d_scores[neighborIdx];

                            // 1. Spatial duplicate (distance <= 1mm): keep only the NanoVDB voxel representative
                            if (distSq <= 0.000001f) {
                                if (neighborScore > currentScore + 1e-6f) {
                                    is_local_max = false;
                                    break;
                                }
                                else if (fabsf(neighborScore - currentScore) <= 1e-5f) {
                                    if (neighborIdx < idx) {
                                        is_local_max = false;
                                        break;
                                    }
                                }
                            }
                            // 2. Inter-voxel NMS comparison (distance > 1mm)
                            else {
                                if (neighborScore > currentScore) {
                                    is_local_max = false;
                                    break;
                                }
                                else if (fabsf(neighborScore - currentScore) <= 1e-6f) {
                                    float3 np = neighbor.pos;
                                    if (np.x < p.x - 1e-3f || 
                                       (fabsf(np.x - p.x) <= 1e-3f && (np.y < p.y - 1e-3f || 
                                       (fabsf(np.y - p.y) <= 1e-3f && (np.z < p.z - 1e-3f || 
                                       (fabsf(np.z - p.z) <= 1e-3f && neighborIdx < idx)))))) {
                                        is_local_max = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Shared memory block-level atomic aggregation
        int localWriteOffset = -1;
        if (is_local_max) {
            localWriteOffset = atomicAdd(&blockCount, 1);
        }
        __syncthreads();

        if (threadIdx.x == 0 && blockCount > 0) {
            blockBaseIdx = atomicAdd(d_keypointCount, blockCount);
        }
        __syncthreads();

        if (is_local_max && localWriteOffset >= 0) {
            int globalOutputIdx = blockBaseIdx + localWriteOffset;
            if (globalOutputIdx < Num_Keypoints_Max) {
                keypoints kp;
                kp.pos = selfPt.pos;
                kp.normal = selfPt.normal;
                kp.normal_trust = selfPt.normal_trust;
                kp.score = currentScore;
                kp.scale = currentScale;
                kp.curvature = selfPt.curvature;
                kp.index_on_map = idx;

                d_outputKeypoints[globalOutputIdx] = kp;
            }
        }
    }

} // namespace roller::loopclosure::cuda