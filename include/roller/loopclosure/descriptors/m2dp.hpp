/*
 *                   江城子 . 程序员之歌
 * 
 *               十年生死两茫茫，写程序，到天亮。
 *                   千行代码，Bug何处藏。
 *               纵使上线又怎样，朝令改，夕断肠。
 * 
 *               领导每天新想法，天天改，日日忙。
 *                   相顾无言，惟有泪千行。
 *               每晚灯火阑珊处，夜难寐，加班狂。
 * 
 * 
 * @Author: Glory Huang
 * @Date: 2021-10-08 09:41:35
 * @LastEditors: Glory Huang
 * @LastEditTime: 2021-10-15 15:59:34
 * @Page: https://xjtuglory.ml
 * @Github: https://github.com/gloryhry
 * @Description: file content
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

class M2DP {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    M2DP(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, double imu_roll = 0.0, double imu_pitch = 0.0);
    ~M2DP() = default;
    Eigen::Matrix<double, 1, 192> get_m2dp_result()
    {
        return m2dp_result;
    }

    Eigen::Matrix<double, 64, 128> get_m2dp_A()
    {
        return A;
    }

private :
    // key parameter
    // number of bins in theta, the 't' in paper
    int numT = 16;
    // number of bins in rho, the 'l' in paper
    int numR = 8;
    // number of azimuth angles, the 'p' in paper
    int numP = 4;
    // number of elevation angles, the 'q' in paper
    int numQ = 16;
    // input pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud{new pcl::PointCloud<pcl::PointXYZ>()};
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered{new pcl::PointCloud<pcl::PointXYZ>()};
    Eigen::MatrixXd cloud_pca;
    // output m2dp result
    Eigen::Matrix<double,1,192> m2dp_result;

    std::vector<double> azimuthList;
    std::vector<double> elevationList;
    double maxRho=0;
    Eigen::Matrix<double, 64, 128> A;

    // main function, get the signature matrix A
    Eigen::Matrix<double, 64, 128> GetSignatureMatrix();

    void cart2sph(double &azm, double &elv, double &r, Eigen::Vector3d vecN);
    void sph2cart(double azm, double elv, double r, Eigen::Matrix<double, 1, 3> &vecN);
    void cart2pol(double x, double y, Eigen::Vector2d &vecN);
    void pol2cart(double rho, double phi, Eigen::Vector2d &vecN);

    Eigen::Vector4f centerCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud);

    void histogram2d(std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> points, std::vector<double> thetaList, std::vector<double> rhoList, Eigen::MatrixXd &hist);
};