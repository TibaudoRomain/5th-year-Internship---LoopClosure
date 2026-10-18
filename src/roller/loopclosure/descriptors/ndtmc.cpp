#include "roller/loopclosure/descriptors/ndtmc.hpp"

#include <cmath>
#include <fstream>
#include <iostream>

NDTMC::NDTMC()
    : pc_in(new pcl::PointCloud<pcl::PointXYZI>),
      ndt_pointcloud(new pcl::PointCloud<pcl::PointXYZI>),
      resolution(1.0f)
{}

NDTMC::NDTMC(float resolution_)
    : pc_in(new pcl::PointCloud<pcl::PointXYZI>),
      ndt_pointcloud(new pcl::PointCloud<pcl::PointXYZI>),
      resolution(resolution_)
{}

void NDTMC::readPCD(const std::string &in_file)
{
    pc_in = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(in_file, *pc_in) == -1)
    {
        std::cerr << "Couldn't read file " << in_file << ".pcd" << std::endl;
        return;
    }
}

std::map<std::size_t, NDTMC::Leaf> NDTMC::readNIOBin(const std::string &in_file)
{
    pc_in = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    std::string lidar_filename_path = in_file;
    std::ifstream inputfile;
    inputfile.open(lidar_filename_path, std::ios::binary);
    if (!inputfile)
    {
        std::cerr << "ERROR: Cannot open file " << lidar_filename_path << "! Aborting..." << std::endl;
        return {};
    }

    std::map<std::size_t, Leaf> leaves_;
    inputfile.seekg(0, std::ios::beg);
    for (std::size_t i = 0; !inputfile.eof(); i++)
    {
        Leaf leaf_;
        inputfile.read(reinterpret_cast<char *>(&leaf_.nr_points), sizeof(int));
        inputfile.read(reinterpret_cast<char *>(&leaf_.mean_), sizeof(Eigen::Vector3d));
        inputfile.read(reinterpret_cast<char *>(&leaf_.cov_), sizeof(Eigen::Matrix3d));
        inputfile.read(reinterpret_cast<char *>(&leaf_.icov_), sizeof(Eigen::Matrix3d));
        inputfile.read(reinterpret_cast<char *>(&leaf_.evecs_), sizeof(Eigen::Matrix3d));
        inputfile.read(reinterpret_cast<char *>(&leaf_.evals_), sizeof(Eigen::Vector3d));

        leaves_.insert(std::make_pair(i, leaf_));
    }
    return leaves_;
}

void NDTMC::readKittiBin(const std::string &in_file)
{
    pc_in = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    std::string lidar_filename_path = in_file;
    std::ifstream inputfile;
    inputfile.open(lidar_filename_path, std::ios::binary);
    if (!inputfile)
    {
        std::cerr << "ERROR: Cannot open file " << lidar_filename_path << "! Aborting..." << std::endl;
        return;
    }

    inputfile.seekg(0, std::ios::beg);
    for (int i = 0; inputfile.good() && !inputfile.eof(); i++)
    {
        pcl::PointXYZI p;
        inputfile.read((char *)&p.x, 3 * sizeof(float));
        inputfile.read((char *)&p.intensity, sizeof(float));
        p.z += LIDAR_HEIGHT;
        pc_in->points.push_back(p);
    }
}

double NDTMC::distDirect(const Eigen::MatrixXd &_frame_sc, Eigen::MatrixXd _submap_sc)
{
    if (_frame_sc.size() == 0 || _submap_sc.size() == 0) return 1.0;

    double a_mean = _frame_sc.mean();
    double b_mean = _submap_sc.mean();

    Eigen::MatrixXd a = _frame_sc.array() - a_mean;
    a = (a.array() == -a_mean).select(0, a);
    Eigen::MatrixXd b = _submap_sc.array() - b_mean;
    b = (b.array() == -b_mean).select(0, b);

    double norm_a = std::sqrt((a.cwiseProduct(a).colwise().sum()).sum());
    double norm_b = std::sqrt((b.cwiseProduct(b).colwise().sum()).sum());

    if (norm_a < 1e-9 || norm_b < 1e-9) {
        return 1.0;
    }

    double sc_sim = (a.cwiseProduct(b).colwise().sum()).sum() / (norm_a * norm_b);
    sc_sim = std::max(-1.0, std::min(1.0, sc_sim));

    return 1.0 - std::abs(sc_sim);
}

Eigen::MatrixXd NDTMC::circshift(Eigen::MatrixXd _mat, const Eigen::MatrixXd &_mat2)
{
    Eigen::MatrixXd shifted_mat = Eigen::MatrixXd::Zero(_mat.rows(), _mat.cols() + _mat2.cols());
    shifted_mat.block(0, 0, _mat.rows(), _mat.cols()) = _mat;
    shifted_mat.block(0, _mat.cols(), _mat.rows(), _mat2.cols()) = _mat.block(0, 0, _mat.rows(), _mat2.cols());
    return shifted_mat;
}

Eigen::MatrixXd NDTMC::circshift(Eigen::MatrixXd _mat)
{
    Eigen::MatrixXd expanded_mat = Eigen::MatrixXd::Zero(_mat.rows(), 2 * _mat.cols() - 1);
    expanded_mat.block(0, 0, _mat.rows(), _mat.cols()) = _mat;
    for (int i = 1; i < _mat.cols(); ++i)
    {
        expanded_mat.block(0, i + _mat.cols() - 1, _mat.rows(), 1) = _mat.block(0, i, _mat.rows(), 1);
    }
    return expanded_mat;
}

std::vector<std::pair<double, int>> NDTMC::distanceBtnNDTScanContext(Eigen::MatrixXd &frame_sc, Eigen::MatrixXd &submap_sc)
{
    std::vector<std::pair<double, int>> results;
    int len_cols = static_cast<int>(submap_sc.cols());

    if (frame_sc.rows() == 0 || submap_sc.rows() == 0 || frame_sc.cols() == 0 || submap_sc.cols() == 0) {
        results.emplace_back(1000.0, -1);
        return results;
    }

    Eigen::MatrixXd submap_sc_ = circshift(submap_sc);
    for (int num_shift = 0; num_shift < len_cols; ++num_shift)
    {
        Eigen::MatrixXd submap_sc_shifted = submap_sc_.block(0, num_shift, frame_sc.rows(), frame_sc.cols());
        double cur_sc_dist = distDirect(frame_sc, submap_sc_shifted);

        if (std::isnan(cur_sc_dist)) {
            continue;
        }

        results.emplace_back(cur_sc_dist, num_shift);
    }

    if (results.empty()) {
        results.emplace_back(1000.0, -1);
    }

    return results;
}

void NDTMC::transformToNDTForm(float resolution_)
{
    if (!pc_in || pc_in->empty()) return;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>(*pc_in);
    voxel_grid_frame.setLeafSize(resolution_, resolution_, resolution_);
    voxel_grid_frame.setInputCloud(cloudPtr);
    voxel_grid_frame.filter(true);
}

void NDTMC::getNDTLeaves(const std::map<std::size_t, Leaf> &leaves_)
{
    ndtcell.clear();
    ndt_pointcloud->clear();
    for (const auto &leave : leaves_)
    {
        if (leave.second.nr_points < 5)
            continue;

        NDTCell temp_(leave.second.mean_, leave.second.cov_);

        pcl::PointXYZI point;
        point.x = static_cast<float>(leave.second.mean_(0));
        point.y = static_cast<float>(leave.second.mean_(1));
        point.z = static_cast<float>(leave.second.mean_(2));
        ndt_pointcloud->points.emplace_back(point);
        ndtcell.emplace_back(temp_);
    }
}

Eigen::MatrixXd NDTMC::getNDTLeaves_new(const std::map<std::size_t, Leaf> &leaves_)
{
    std::vector<std::vector<std::vector<GridCell>>> grid(
        PC_NUM_RING, std::vector<std::vector<GridCell>>(
            PC_NUM_SECTOR, std::vector<GridCell>(PC_NUM_Z)));

    Eigen::MatrixXd desc_1(PC_NUM_RING, PC_NUM_SECTOR);
    desc_1.setZero();
    histogram.setZero();
    for (const auto &leave : leaves_)
    {
        if (leave.second.nr_points < 5 || leave.second.evals_(2) == 0){
            continue;
        }
        Eigen::Matrix3d cov = leave.second.cov_;
        pcl::PointXYZI point;
        point.x = static_cast<float>(leave.second.mean_(0));
        point.y = static_cast<float>(leave.second.mean_(1));
        point.z = static_cast<float>(leave.second.mean_(2));
        int ring_idx, sector_idx;
        if (!CalculateSectorId(point, ring_idx, sector_idx))
        {
            continue;
        }
        int z_idx = CalculateLayer(point.z);
        if (z_idx == -1)
        {
            continue;
        }

        double value = leave.second.evals_(2) * leave.second.evals_(0) / (leave.second.evals_(1) * leave.second.evals_(1));
        
        double interval_size = 3.0 / double(num_intervals);
        if (value < 3) {
            int his_idx = static_cast<int>(std::floor(value / interval_size));
            if (his_idx >= 0 && his_idx < num_intervals) {
                histogram(0, his_idx)++;
            }
        }

        if (value > 0 && value < 2.4) {
            int shape_id = static_cast<int>(std::floor(value / 0.3));
            grid[ring_idx - 1][sector_idx - 1][z_idx - 1].addShape(shape_id + 1);
            double N = 3.0;
            desc_1(ring_idx - 1, sector_idx - 1) += ((N / 2) * (1 + log(2 * M_PI)) + 0.5 * log(cov.determinant())) * z_idx / PC_NUM_Z;
        } else {
            continue;
        }
    }

    Eigen::MatrixXd desc_2(PC_NUM_RING, PC_NUM_SECTOR);
    desc_2.setZero();
    for (int i = 0; i < PC_NUM_RING; i++)
    {
        for (int j = 0; j < PC_NUM_SECTOR; j++)
        {
            double weight = 0;
            for (int k = 0; k < PC_NUM_Z; k++)
            {
                if (grid[i][j][k].shape_max == -1) {
                    continue;
                }
                weight += double(grid[i][j][k].shape_max * (k + 1)) / PC_NUM_Z;
            }
            desc_2(i, j) = weight;
        }
    }

    Eigen::MatrixXd desc(2 * PC_NUM_RING, PC_NUM_SECTOR);
    desc << desc_1,
            desc_2;
    return desc;
}

float NDTMC::xy2theta(const float &_x, const float &_y)
{
    if ((_x >= 0) && (_y >= 0))
        return static_cast<float>((180.0 / M_PI) * atan(_y / _x));

    if ((_x < 0) && (_y >= 0))
        return static_cast<float>(180.0 - ((180.0 / M_PI) * atan(_y / (-_x))));

    if ((_x < 0) && (_y < 0))
        return static_cast<float>(180.0 + ((180.0 / M_PI) * atan(_y / _x)));

    if ((_x >= 0) && (_y < 0))
        return static_cast<float>(360.0 - ((180.0 / M_PI) * atan((-_y) / _x)));
    return 0.0f;
}

bool NDTMC::CalculateSectorId(const pcl::PointXYZI &pt_in, int &ring_idx, int &sctor_idx)
{
    float azim_range = std::sqrt(pt_in.x * pt_in.x + pt_in.y * pt_in.y);
    float azim_angle = xy2theta(pt_in.x, pt_in.y);
    if (azim_range > PC_MAX_RADIUS)
        return false;

    ring_idx = std::max(std::min(PC_NUM_RING, int(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), 1);
    sctor_idx = std::max(std::min(PC_NUM_SECTOR, int(ceil((azim_angle / 360.0) * PC_NUM_SECTOR))), 1);
    return true;
}

int NDTMC::CalculateLayer(float z) const
{
    if (z < 0 || z > PC_MAX_Z)
    {
        return -1;
    }
    int layer = std::max(std::min(PC_NUM_Z, int(ceil((z / PC_MAX_Z) * PC_NUM_Z))), 1);
    return layer;
}

int NDTMC::GetDistributionWeight(int idx)
{
    if (idx < 0 || idx >= static_cast<int>(ndtcell.size())) return -1;
    NDTCell* cell_temp = &ndtcell[idx];
    Eigen::Matrix3d cov = cell_temp->cov_;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov);
    double value = -1.0;
    if (saes.eigenvalues()(2) < 0.001 * saes.eigenvalues()(0))
    {
        value = 0.001 * saes.eigenvalues()(0) * saes.eigenvalues()(0) /
                (saes.eigenvalues()(1) * saes.eigenvalues()(1));
    }
    else
    {
        value = saes.eigenvalues()(2) * saes.eigenvalues()(0) /
                (saes.eigenvalues()(1) * saes.eigenvalues()(1));
    }

    double interval_size = 3.0 / double(num_intervals);
    if (value < 3)
    {
        int his_idx = static_cast<int>(std::floor(value / interval_size));
        if (his_idx >= 0 && his_idx < num_intervals) {
            histogram(0, his_idx)++;
        }
    }

    if (value > 0 && value < 2.4)
    {
        int shape_id = static_cast<int>(std::floor(value / 0.3));
        return shape_id + 1;
    }

    return -1;
}

int NDTMC::CalculateShapeWeight(const GridCell &cell)
{
    return cell.shape_max > 0 ? cell.shape_max : 0;
}

Eigen::MatrixXd NDTMC::createNDTMC_KITTI()
{
    std::vector<std::vector<std::vector<GridBin>>> grid(
        PC_NUM_RING, std::vector<std::vector<GridBin>>(
            PC_NUM_SECTOR, std::vector<GridBin>(PC_NUM_Z)));

    for (int idx = 0; idx < static_cast<int>(pc_in->size()); ++idx)
    {
        pcl::PointXYZI pt = pc_in->points[idx];
        int ring_idx, sector_idx;
        if (!CalculateSectorId(pt, ring_idx, sector_idx))
        {
            continue;
        }
        int z_idx = CalculateLayer(pt.z);
        if (z_idx == -1)
        {
            continue;
        }
        grid[ring_idx - 1][sector_idx - 1][z_idx - 1].addPoint(pt.x, pt.y, pt.z);
    }

    Eigen::MatrixXd desc(PC_NUM_RING, PC_NUM_SECTOR);
    desc.setZero();
    for (int k = 0; k < PC_NUM_Z; ++k)
    {
        for (int i = 0; i < PC_NUM_RING; ++i)
        {
            for (int j = 0; j < PC_NUM_SECTOR; ++j)
            {
                if (grid[i][j][k].num_of_points < 3)
                    continue;
                Eigen::Matrix3d cov = grid[i][j][k].cov;
                if (cov.norm() > 10)
                    continue;
                double N = 4.0;
                desc(i, j) += (N / 2) * (1 + log(2 * M_PI)) + 0.5 * log(cov.determinant());
            }
        }
    }
    return desc;
}

Eigen::MatrixXd NDTMC::createNDTMC()
{
    transformToNDTForm(resolution);
    return getNDTLeaves_new(voxel_grid_frame.getLeaves());
}

Eigen::MatrixXd NDTMC::createNDTMC(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud_in)
{
    if (!cloud_in || cloud_in->empty()) return Eigen::MatrixXd();
    pcl::copyPointCloud(*cloud_in, *this->pc_in);

    // Center point cloud to (0,0,0) so polar coordinate radiuses fit within PC_MAX_RADIUS
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*this->pc_in, centroid);
    for (auto& pt : this->pc_in->points) {
        pt.x -= centroid[0];
        pt.y -= centroid[1];
        pt.z -= centroid[2];
    }

    return createNDTMC();
}

Eigen::MatrixXd NDTMC::createNDTMC(const std::map<std::size_t, NDTMC::Leaf> &leaves)
{
    return getNDTLeaves_new(leaves);
}
