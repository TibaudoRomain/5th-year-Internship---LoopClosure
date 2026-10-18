# ROLLER: 3D Acoustic SLAM & Global Loop Closure

A C++/CUDA ROS 2 framework for 3D subsea mapping and real-time global loop closure using multibeam forward-looking sonar (Water Linked Sonar 3D-15). Developed at the **Ocean Systems Laboratory (OSL)**, Heriot-Watt University, in partnership with **Polytech Nice Sophia**.

---

## 1. System Overview

Underwater navigation relies primarily on dead reckoning from Doppler Velocity Logs (DVL) and Inertial Navigation Systems (INS). Over extended trajectories, dead reckoning accumulates drift, creating duplicate walls and deformed geometry in the reconstructed map. 

This repository provides a decoupled two-tier SLAM system with a submap-based 6-DOF loop closure engine:

```
[3D Sonar Returns] 
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Synchronous Frontend (5–20 Hz)                           │
│    ├── Point cloud scan-matching odometry (fast_gicp)       │
│    ├── Local keyframe pose graph optimization               │
│    └── Volumetric TSDF submap integration (NanoVDB)         │
└──────────────────────────────┬──────────────────────────────┘
                               │ Finalized Submap
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Asynchronous Backend (Submap Rate)                       │
│    ├── Submap database storage                              │
│    ├── GPU surface normal calculation (3D Sobel)            │
│    ├── CUDA 3D keypoint detection (Harris3D / ISS3D + NMS)  │
│    ├── Feature description (FPFH, SHOT, HPNVD, R-FPFH)      │
│    ├── Candidate gating & Yaw prior search (NDTMC)          │
│    ├── 6-DOF Registration (TEASER++ / Umeyama)              │
│    └── Global factor graph smoothing (GTSAM / iSAM2)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Key Modules & Pipeline

### A. Pre-processing & Surface Normal Estimation
- **TSDF Zero-Crossing Extraction**: Ray-traces the sensor beam into the TSDF grid to reconstruct the implicit surface.
- **Sobel Surface Normals (GPU)**: Evaluates $n = -\frac{\nabla\phi}{\|\nabla\phi\|}$ directly from the voxel field on CUDA.
- **Normal Smoothing**: Optional GPU Gaussian normal smoothing and multipath grazing-angle rejection ($\theta > 80^\circ$).

### B. 3D Keypoint Detection
- **GPU (CUDA)**:
  - `harris3d_cuda`: Local covariance tensor curvature response evaluated in parallel.
  - `iss3d_cuda`: Intrinsic Shape Signatures measuring point dispersion eigenvalue ratios ($\lambda_3 / \lambda_1$).
  - **Scale-Awareness & 3D NMS**: Multi-scale radius evaluation and 3D spatial Non-Maximum Suppression to ensure uniform coverage across the submap.
- **CPU (PCL Fallback)**:
  - `harris3d`, `iss`, or uniform `voxelisation`.

### C. Feature Descriptors
- **Local Descriptors**:
  - `fpfh`: 33-dimensional Fast Point Feature Histograms encoding angular normal deviations.
  - `shot`: 352-dimensional Signature of Histograms of Orientations.
  - `hpnvd`: 324-dimensional Histogram of Point and Normal Vector Distributions.
  - `r-fpfh`: 44-dimensional reflectivity-augmented descriptor fusing geometric FPFH with normalized acoustic intensity statistics.
- **Global Place Recognition**:
  - `ndtmc`: Cylindrical Normal Distributions Transform Monte Carlo / Scan Context matrix used for coarse candidate identification and rotational yaw prior estimation.

### D. Candidate Gating & Matching
- **4-Tier Candidate Filtering**:
  1. Temporal age exclusion (avoids matching consecutive submaps).
  2. Trajectory topological proximity.
  3. Covariance drift ellipse gating.
  4. Global descriptor distance thresholding.
- **Feature Matching**:
  - Mutual Nearest Neighbors (MNN / Reciprocal check).
  - Lowe's Ratio test.
  - Sinkhorn optimal transport assignment.

### E. 6-DOF Transformation Estimation
- **TEASER++**: Fast certifiable registration using Truncated Least Squares, Translation Invariant Measurements (TIMs), and Maximum Clique inlier formulation. Robust to $>90\%$ outlier ratios.
- **Umeyama with Normals**: Closed-form SVD registration incorporating surface normal compatibility inside RANSAC.

---

## 3. Directory Layout

```
Project2_loopclosure/
├── CMakeLists.txt              # CMake configuration (ament_cmake, CUDA separable compilation)
├── package.xml                 # ROS 2 package definition
├── config/
│   ├── calibration/            # Sensor-to-vehicle extrinsic calibrations
│   ├── frames/                 # TF frame definitions
│   ├── rviz/                   # RViz display configurations
│   └── slam/
│       ├── loopclosure.yaml    # Main loop closure parameters (keypoints, descriptors, solvers)
│       ├── sonar3d.yaml        # Sonar node parameters
│       └── sonar3d_core.yaml   # SLAM frontend parameters
├── include/roller/
│   ├── base_node.hpp           # Base ROS 2 SLAM node class
│   ├── submap.hpp              # Submap and Keyframe data structures
│   ├── tsdf/                   # NanoVDB TSDF integration headers
│   └── loopclosure/
│       ├── Config.hpp          # Runtime configuration manager and variant definitions
│       ├── host_pipeline.hpp   # Host-side loop closure coordinator (loopclosure_master)
│       ├── benchmark.hpp       # Evaluation and benchmarking harness (RTE, RRE, timing)
│       ├── pcl_toolbox.hpp     # PCL utility functions
│       ├── octree.hpp          # Spatial indexing structures
│       ├── descriptors/        # NDTMC, HPNVD, and M2DP implementations
│       └── cuda/               # CUDA pipeline headers and wrapper interfaces
├── src/
│   ├── base_node.cpp           # Odometry handling and topic synchronization
│   ├── sonar3d_node.cpp        # Standalone Sonar 3D SLAM executable
│   ├── roller_node.cpp         # Main multi-sensor SLAM node
│   └── roller/
│       ├── backend/            # Factor graph optimization and submap management
│       ├── frontend/           # Scan matching and TSDF generation
│       └── loopclosure/        # Loop closure source files
│           ├── host_pipeline.cpp  # Pipeline orchestration, candidate gating, TEASER++
│           ├── benchmark.cpp      # Logging of execution metrics and alignment errors
│           └── cuda/              # CUDA kernels (normals.cu, keypoints.cu, descriptors.cu)
├── launch/
│   ├── sonar3d_launch.py       # Full Sonar 3D SLAM with loop closure
│   ├── sonar3d_standalone_launch.py
│   └── roller_replay_launch.py # Rosbag replay launch
└── srv/
    └── StringTrigger.srv       # Service interface for pipeline control
```

---

## 4. Dependencies & Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill or newer
- **CUDA**: 11.8+ / 12.x (Compute Capability 6.1, 7.5, 8.6, 8.9)
- **Compiler**: GCC 11+ with C++17 support

### Third-Party Libraries
| Library | Minimum Version | Description |
|:---|:---|:---|
| **PCL** | 1.12+ | Point Cloud Library (filters, features, KD-Tree) |
| **OpenVDB / NanoVDB** | 9.0+ | Sparse hierarchical voxel data structure |
| **GTSAM** | 4.2+ | Factor graph optimization library (iSAM2) |
| **TEASER++** | Latest | Fast and certifiable 3D point cloud registration |
| **fast_gicp** | Latest | Fast Generalized ICP for frontend odometry |
| **OpenCV** | 4.5+ | Image utilities and matrix operations |
| **Eigen3** | 3.4+ | Linear algebra and geometric transformations |
| **OctoMap** | 1.9+ | Spatial occupancy mapping |

---

## 5. Build Instructions

Clone or place the package inside your ROS 2 workspace `src/` directory:

```bash
cd ~/ros2_ws/src
# Ensure Project2_loopclosure is located as roller/

cd ~/ros2_ws
# Install ROS dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build with CUDA separable compilation and release flags
colcon build --symlink-install --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCHITECTURES="75;86"
```

Source the workspace overlay:
```bash
source ~/ros2_ws/install/setup.bash
```

---

## 6. Configuration

All loop closure parameters are defined in [`config/slam/loopclosure.yaml`](config/slam/loopclosure.yaml).

### Key Parameters:

```yaml
sonar3d_node:
  ros__parameters:
    # 1. Surface Normal Estimation
    normal_calculation:
      gaussian_filter: true
      gaussian_width: 2
      gaussian_repetitions: 3
      distance_to_tsdf: 0.3

    # 2. Keypoint Selection
    estimate_keypoints:
      method: "harris3d_cuda"     # "harris3d_cuda", "iss3d_cuda", "harris3d", "iss"
      filtering_method: "nms"     # "nms", "top_k", "thresholding"
      nms:
        radius: 0.5               # 3D NMS radius in meters
      harris3d_cuda:
        score_threshold: -0.0325
        k: 0.04

    # 3. Descriptor Extraction
    extract_feature_descriptors:
      method: "fpfh"              # "fpfh", "shot", "hpnvd"
      fpfh:
        use_k_instead_of_radius: false
        fpfh_radius: 1.5
        fpfh_k: 20

    # 4. Feature Matching
    match_features:
      method: "base_pcl"          # "base_pcl", "Lowe_Ratio", "Sinkhorn"
      lowe_ratio: 0.8
      reciprocal: true

    # 5. Transformation Solver
    transformation_estimation:
      method: "teaser"            # "teaser" or "umeyama"
      teaser:
        estimate_scaling: false
        noise_bound_multiplier: 5.0
        rotation_estimation_algorithm: "GNC_TLS"
        inlier_selection_mode: "PMC_EXACT"
        max_clique_time_limit: 0.5
```

---

## 7. Execution & Launch

### Launch SLAM with 3D Sonar & Loop Closure:
```bash
ros2 launch roller sonar3d_launch.py
```

### Replay from Rosbag:
```bash
ros2 launch roller roller_replay_launch.py
```

### Published Debug & Visualization Topics (RViz):
- `/loopclosure/new_submap_keypoints` (`sensor_msgs/PointCloud2`): Keypoints extracted from the current query submap.
- `/loopclosure/old_submap_keypoints` (`sensor_msgs/PointCloud2`): Keypoints from the candidate reference submap.
- `/loopclosure/new_submap_cloud` (`sensor_msgs/PointCloud2`): Zero-crossing surface mesh of query submap.
- `/loopclosure/old_submap_cloud` (`sensor_msgs/PointCloud2`): Zero-crossing surface mesh of candidate submap.
- `/loopclosure/transformed_cloud` (`sensor_msgs/PointCloud2`): Query submap aligned onto candidate frame via estimated transform.
- `/loopclosure/correspondences` (`visualization_msgs/Marker`): Feature correspondence lines (green for inliers, red for rejected outliers).

---

## 8. Benchmarks & Performance Summary

Evaluated on an NVIDIA RTX 3080 / Intel i7-11800H platform across volumetric submaps containing $>100\text{k}$ voxels:

| Pipeline Stage | CPU (PCL / OpenMP) | GPU (CUDA / NanoVDB) |
|:---|:---:|:---:|
| **Surface Normal Calculation** | 162.65 ms | **4.48 ms** |
| **Normal Smoothing** | — | **0.92 ms** |
| **3D Harris Keypoints** | 3,120.00 ms | **3.91 ms** |
| **ISS 3D Keypoints** | 1,787.29 ms | **7.54 ms** |
| **Tensor Omnivariance Saliency** | 18,508.63 ms | **57.17 ms** |
| **Candidate Search (NDTMC)** | 1.20 ms | — |
| **6-DOF Solver (TEASER++)** | 10–30 ms | — |
| **Factor Graph Update (iSAM2)** | < 8.5 ms | — |

---

## 9. Citation & Contact

- **Author**: Romain TIBAUDO (`romain.tibaudo@etu.unice.fr`)
- **Supervisors**: Simon Archieri, Prof. Yvan Petillot (Ocean Systems Laboratory, Heriot-Watt University)
- **Academic Referent**: Pascal Masson (Polytech Nice Sophia)
