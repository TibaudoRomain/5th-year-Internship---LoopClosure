#pragma once
#include <Eigen/Dense>

#include <random>
#include <execution>

#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/common/centroid.h>

namespace roller::loopclosure {
    struct TransformationError {
        float translation_error;
        float rotation_error;
    };
    
    namespace Utils {
        inline std::mt19937& get_rng() {
            // thread_local pour être "thread-safe" sans mutex
            static thread_local std::random_device rd;
            static thread_local std::mt19937 gen(rd());
            return gen;
        }

        static float getRandom(float min, float max) {
            std::uniform_real_distribution<float> dist(min, max);
            return dist(get_rng());
        }
        
        template <typename Derived>
        inline std::string matrix_to_string(const Eigen::MatrixBase<Derived>& matrix) {
            std::stringstream ss;
            ss << std::endl << matrix;
            return ss.str();
        }

        template <typename PointT>
        inline Eigen::Vector4f get_centroid(const pcl::PointCloud<PointT>& cloud){
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(cloud, centroid);
            return centroid;
        }



        inline TransformationError transformation_matrix_eval(
                Eigen::Ref<const Eigen::Matrix4f> goal, 
                Eigen::Ref<const Eigen::Matrix4f> to_evaluate)
        {
            TransformationError evaluation;
            
            Eigen::Vector3f t_goal = goal.block<3, 1>(0, 3);
            Eigen::Vector3f t_eval = to_evaluate.block<3, 1>(0, 3);

            evaluation.translation_error = (t_goal - t_eval).norm();

            Eigen::Matrix3f R_goal = goal.block<3, 3>(0, 0);
            Eigen::Matrix3f R_eval = to_evaluate.block<3, 3>(0, 0);

            Eigen::Matrix3f R_relative = R_goal.transpose() * R_eval;

            float trace = R_relative.trace();
            float cos_theta = (trace - 1.0f) / 2.0f;

            cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
            
            float angle_rad = std::acos(std::clamp(cos_theta, -1.0f, 1.0f));
            evaluation.rotation_error = angle_rad * (180.0f / M_PI);

            return evaluation;
        }
    }

    namespace Noise {
        template <typename PointT>
        void addGaussianNoise(pcl::PointCloud<PointT>& cloud, float standard_deviation = 0.01f){
            if(cloud.empty()){return;}
            std::normal_distribution<float> nd(0.0f, standard_deviation);
            std::for_each(std::execution::par_unseq , cloud.begin(), cloud.end(), [&](PointT& point){
                auto& rng = Utils::get_rng();
                point.x +=  nd(rng);
                point.y += nd(rng);
                point.z += nd(rng);
            });
        }

        template <typename PointT>
        void addUniformNoise(pcl::PointCloud<PointT>& cloud, float min_offset, float max_offset) {
            if (cloud.empty()) {return;}
            std::uniform_real_distribution<float> ud(min_offset, max_offset);

            std::for_each(std::execution::par_unseq , cloud.begin(), cloud.end(), [&](PointT& point){
                auto& rng = Utils::get_rng();
                point.x += ud(rng);
                point.y += ud(rng);
                point.z += ud(rng);
            });
        }

        template <typename PointPtrT>
        inline auto subSamplePointCloud(
                const PointPtrT& cloud, 
                float min_space_between_points){
            using PointT = typename PointPtrT::element_type::PointType;
            using CloudPtr = typename pcl::PointCloud<PointT>::Ptr;
            if (!cloud || cloud->empty()) {
                return CloudPtr(); 
            }
            
            CloudPtr filtered_cloud= std::make_shared<pcl::PointCloud<PointT>>();

            pcl::UniformSampling<PointT> us;
            us.setInputCloud(cloud);
            us.setRadiusSearch(min_space_between_points);
            us.filter(*filtered_cloud);

            return filtered_cloud;
        }

    }
    /*namespace geom {
        struct IndexCoord {
            IntCoord(int32_t _x = 0, int32_t _y = 0, int32_t _z = 0) : x(_x), y(_y), z(_z) {}
            
            template <typename Vdb>
            explicit IntCoord(const typename Vdb::Coord& vdbCoord) : x(vdbCoord.x()), y(vdbCoord.y()), z(vdbCoord.z()) {}
            
            int32_t x;
            int32_t y;
            int32_t z;

            template <typename Vdb>
            static IntCoord from_vdb(const typename Vdb::Coord& vdbCoord) {
                return IntCoord(vdbCoord);
            }

            openvdb::Coord to_open() const {return openvdb::Coord(x, y, z);}
            nanovdb::Coord to_nano() const {return nanovdb::Coord(x, y, z);}
        };
        struct WorldCoord {
            float x;
            float y;
            float z;
        };

        struct IndexCoord {
            IntCoord(int32_t _x = 0, int32_t _y = 0, int32_t _z = 0) : x(_x), y(_y), z(_z) {}
            
            template <typename Vdb>
            explicit IntCoord(const typename Vdb::Coord& vdbCoord) : x(vdbCoord.x()), y(vdbCoord.y()), z(vdbCoord.z()) {}
            
            float x;
            float y;
            float z;

            template <typename Vdb>
            static IntCoord from_vdb(const typename Vdb::Coord& vdbCoord) {
                return IntCoord(vdbCoord);
            }

            openvdb::Coord to_open() const {return openvdb::Coord(x, y, z);}
            nanovdb::Coord to_nano() const {return nanovdb::Coord(x, y, z);}
        };
        
    }*/
}