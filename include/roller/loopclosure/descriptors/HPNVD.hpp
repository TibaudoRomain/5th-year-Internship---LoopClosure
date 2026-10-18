#pragma once

#include <vector>
#include <numeric>
#include <cmath>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

// PCL Point Struct for HPNVD 324-dimensional Descriptor
struct EIGEN_ALIGN16 HPVNDSignature324 {
    float histogram[324];
    static int descriptorSize() { return 324; }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(HPVNDSignature324,
    (float[324], histogram, hpvnd)
)

namespace pcl {
    template <>
    class DefaultPointRepresentation<HPVNDSignature324> : public PointRepresentation<HPVNDSignature324> {
    public:
        DefaultPointRepresentation() {
            nr_dimensions_ = 324;
        }

        virtual void copyToFloatArray(const HPVNDSignature324 &p, float *out) const override {
            for (int i = 0; i < nr_dimensions_; ++i) {
                out[i] = p.histogram[i];
            }
        }
    };
}

typedef struct {
	float x;
	float y;
	float z;
} Vertex;

typedef struct {
	int pointID;
	Vertex x_axis;
	Vertex y_axis;
	Vertex z_axis;
} LRF;

void HPVND_LRF_Z_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex& z_axis);
void HPVND_LRF_X_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex z_axis, float sup_radius, std::vector<float> PointDist, Vertex& x_axis);
void HPVND_LRF_Y_axis(Vertex x_axis, Vertex z_axis, Vertex& y_axis);
void HPVND_LRF_for_cloud_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<int> indices, float sup_radius, std::vector<LRF>& Cloud_LRF);
void transformCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, LRF pointLRF, pcl::PointCloud<pcl::PointXYZ>::Ptr& transformed_cloud);
void local_HPVND(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float sup_radius, int bin, std::vector<float>& histogram);

void HPVND_compute(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	std::vector<int> indices,
	float sup_radius,
	int bin_num,
	pcl::PointCloud<HPVNDSignature324>::Ptr& Histograms
);
