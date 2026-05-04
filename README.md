# Semantic 3D Object Localization Pipeline

This repository performs **zero-shot 3D object localization** using:

* Multi-view RGB images
* Camera poses + intrinsics
* 3D Gaussian-based reconstruction
* Depth-guided 2D → 3D lifting
* Geometry-aware filtering

The final output is an **Oriented Bounding Box (OBB)** for a queried object.

---

# ⚙️ Requirements

* CUDA **11.8**
* Python **3.10**
* PyTorch **2.5.1 (CUDA 11.8 compatible)**

Install PyTorch:

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📁 Input Data

```
Data/
├── images/          # multi-view RGB images
├── poses.json       # camera-to-world matrices
├── intrinsic.json   # camera intrinsics
```

### Formats

* `images/` → standard RGB images
* `poses.json` → dict of 4×4 camera pose matrices
* `intrinsic.json` → 3×3 camera intrinsic matrix

---

# 🧱 Step 1: Convert to ScanNet Format

```bash
python prep_scannet_data.py
```

---

# 🧭 Step 2: Run COLMAP (Sparse Reconstruction)

Convert inputs:

```bash
python write_colmap_intrinsics.py \
    --images Data/images \
    --poses Data/poses.json \
    --intrinsics Data/intrinsic.json \
    --output model
```

Run COLMAP:

```bash
colmap database_creator --database_path database.db

colmap feature_extractor \
    --database_path database.db \
    --image_path Data \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.camera_params "1477.0097,1480.4424,1298.25,686.82" \
    --ImageReader.single_camera 1

colmap exhaustive_matcher --database_path database.db

colmap model_converter \
    --input_path model \
    --output_path model_bin \
    --output_type BIN

colmap point_triangulator \
    --database_path database.db \
    --image_path Data/images \
    --input_path model_bin \
    --output_path triangulated

colmap model_converter \
    --input_path triangulated \
    --output_path tri_txt \
    --output_type TXT
```

---

# 🌐 Step 3: Semantic Gaussians Pipeline

```bash
cd third_party
git clone <semantic-gaussians-repo>
cd semantic-gaussians
pip install -r requirements.txt
```

### Run:

```bash
python train.py
python fusion.py
python distill.py
```

### Output:

```
outputs/point_cloud.ply
```

---

# 🧠 Step 4: Semantic 2D → 3D Lifting

### Dependencies

* GroundingDINO
* SAM2
* Depth Anything

(Place under `third_party/` and add to `PYTHONPATH`)

---

### Run Query

```bash
python3 run_semantic_lift.py \
    --text-query "vga_socket" \
    --image-dir Data/images \
    --poses-path Data/poses.json \
    --intrinsics-path Data/intrinsic.json \
    --ply-path outputs/reconstruction/point_cloud.ply \
    --output-json outputs/vga_socket_result.json
```

---

# 🔬 Method Overview

The localization pipeline follows a **geometry-aware refinement strategy**:

### 1. Depth-Guided Lifting

* Segmentation masks are projected into 3D using depth + intrinsics
* Subsampling ensures efficiency and robustness

### 2. Nearest Neighbor Projection

* Lifted points are snapped to the reconstructed point cloud
* Reduces depth noise and aligns with scene geometry

### 3. Center Refinement

* A stable object center is computed from matched 3D points
* Improves localization accuracy over raw lifted points

### 4. Adaptive Spatial Gating

* Points are filtered using distance percentiles (not fixed thresholds)
* Ensures scale robustness across scenes

### 5. Planar Background Removal

* PCA is used to detect dominant planes
* Removes large background surfaces (e.g., walls, panels)

### 6. Final OBB Estimation

* Outliers are removed via radial filtering
* OBB is computed using PCA
* Center is stabilized using geometry-aware estimation

---

# 📦 Output

Example:

```json
{
  "entity": "vga_socket",
  "obb": {
    "center": [...],
    "extent": [...],
    "rotation": [...]
  }
}
```

# 🚀 Notes

* IoU is computed via **2D projection of the 3D OBB**
* Slightly larger extents can improve IoU stability
* Center accuracy is more critical than rotation precision
* Percentile-based filtering avoids brittle tuning

---

# 🧘 Summary

```
Images + Poses + Intrinsics
        ↓
COLMAP Reconstruction
        ↓
3D Gaussian Representation
        ↓
2D Segmentation + Depth Lifting
        ↓
Geometry-Aware Filtering
        ↓
Final OBB
```
