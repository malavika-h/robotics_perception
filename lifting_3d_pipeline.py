"""
End-to-end 2D-to-3D Semantic Lifting Pipeline

Detection → Segmentation → Feature Extraction → 3D Lifting
Uses Grounding DINO, SAM2, CLIP, and camera geometry to lift semantic features
from 2D images to 3D point clouds.
"""

from logging import config
import os

import numpy as np
import json
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from third_party.depth_anything.metric_depth.zoedepth.utils import config
import warnings
from PIL import Image
from typing import Tuple, Dict, Any, Optional
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import torch
from sklearn.neighbors import NearestNeighbors

# Try to import required libraries
try:
    from groundingdino.models import build_model
    from groundingdino.util.inference import load_image, predict
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util.slconfig import SLConfig
    import groundingdino.datasets.transforms as T
except ImportError:
    warnings.warn("GroundingDINO not found. Install with: pip install groundingdino-hf")

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    warnings.warn("SAM2 not found. Ensure third_party/sam2 is on PYTHONPATH")

try:
    import open_clip
except ImportError:
    warnings.warn("OpenCLIP not found. Install with: pip install open_clip_torch")

try:
    import trimesh
except ImportError:
    warnings.warn("trimesh not found. Install with: pip install trimesh")

import os
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
config_dir = os.path.abspath("third_party/sam2/sam2/configs/sam2.1")

initialize_config_dir(
    config_dir=config_dir,
    job_name="sam2.1"
)

@dataclass
class PipelineConfig:
    """Configuration for the lifting pipeline."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grounding_dino_config: str = "config/GroundingDINO_SwinB_cfg.py"
    grounding_dino_checkpoint: str = "weights/groundingdino_swinb_cogcoor.pth"
    sam2_checkpoint: str = "third_party/sam2/sam2.1_hiera_large.pt"
    sam2_config: str = "sam2.1_hiera_l"
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    grid_size: int = 8
    distance_threshold: float = 0.15
    percentile_outlier: float = 90.0
    min_points: int = 20
    dino_box_threshold: float = 0.3
    dino_text_threshold: float = 0.25
    use_nearest_neighbor: bool = True
    nn_k: int = 10


class GroundingDINODetector:
    """Detect objects using Grounding DINO."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        print(f"Loading GroundingDINO from {config.grounding_dino_checkpoint}")
        
        # Load model
        cfg = SLConfig.fromfile(config.grounding_dino_config)
        self.model = build_model(cfg)

        checkpoint = torch.load(config.grounding_dino_checkpoint, map_location=self.device)
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model = self.model.to(self.device).eval()

    def detect(
            self, 
            image: np.ndarray, 
            text_query: str
        ) -> List[Tuple[float, float, float, float]]:

        # 1. Ensure the image is in RGB for PIL
        # Your snippet already has a BGR check; let's make it robust
        image_pil = Image.fromarray(image).convert("RGB")

        # 2. Manual Preprocessing (What load_image usually does)
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)

        # 3. Predict
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text_query,
            box_threshold=self.config.dino_box_threshold,
            text_threshold=self.config.dino_text_threshold,
            device=self.device,
            remove_combined=True
        )

        # 4. Convert format (GroundingDINO returns [cx, cy, w, h] normalized)
        detections = []
        for box in boxes:
            # box is [cx, cy, w, h] in normalized [0, 1] coordinates
            # Convert to [x1, y1, x2, y2] if your lifter expects corners
            cx, cy, w, h = box.cpu().numpy()
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            detections.append((float(x1), float(y1), float(x2), float(y2)))

        return detections


class SAM2Segmenter:
    """Segment objects using SAM2."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        print(f"Loading SAM2 from {config.sam2_checkpoint}")
        
        self.model = build_sam2(
            config.sam2_config,        # FIRST positional arg
            config.sam2_checkpoint,    # SECOND
            device=self.device         # optional keyword
        )
        self.predictor = SAM2ImagePredictor(self.model)
    
    def segment(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Segment object in image using bounding box prompt.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            bbox: Normalized bounding box [x1, y1, x2, y2] in [0, 1]
        
        Returns:
            Binary mask (H, W) with 1s for object, 0s for background
        """
        # Convert to RGB if needed
        if image.dtype == np.uint8 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Set image
        self.predictor.set_image(image_rgb)
        
        # Convert normalized bbox to pixel coordinates
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        # box = np.array([x1 * w, y1 * h, x2 * w, y2 * h])
        box = np.array([x1, y1, x2, y2])
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            box=box[np.newaxis, :],
            multimask_output=False
        )
        
        # Return highest confidence mask
        mask = masks[0, 0].astype(np.uint8)
        return mask


class CLIPFeatureExtractor:
    """Extract semantic features using CLIP."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        print(f"Loading CLIP model {config.clip_model}")
        
        self.model, _, self.image_preprocess = open_clip.create_model_and_transforms(
            config.clip_model,
            pretrained=config.clip_pretrained,
            device=self.device
        )
        self.model = self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(config.clip_model)

    def extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract CLIP features from entire image.
        
        Args:
            image: Image as numpy array (H, W, 3). 
                Expected to be BGR if coming from cv2.imread.
        
        Returns:
            Feature vector (feature_dim,) normalized to unit length
        """
        # 1. Convert BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        
        # 2. Convert NumPy array to PIL Image (Required by torchvision transforms)
        # This fixes the "TypeError: Unexpected type <class 'numpy.ndarray'>"
        image_pil = Image.fromarray(image_rgb.astype('uint8'))
        
        # 3. Preprocess and move to device
        image_tensor = self.image_preprocess(image_pil).unsqueeze(0).to(self.device)
        
        # 4. Extract features
        with torch.no_grad():
            # Using autocast for efficiency if your lab machine supports it
            with torch.cuda.amp.autocast():
                features = self.model.encode_image(image_tensor)
        
        # 5. Normalize to unit length
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features[0].detach().cpu().numpy()
    
    def extract_masked_features(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        method: str = "crop"
    ) -> np.ndarray:
        """
        Extract CLIP features from masked region.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            mask: Binary mask (H, W) with 1s for region of interest
            method: "crop" (extract bbox) or "inpaint" (use mask directly)
        
        Returns:
            Feature vector (feature_dim,) normalized to unit length
        """

        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        
        # Squeeze out any extra dimensions (e.g., [1, 1, H, W] -> [H, W])
        mask = mask.squeeze()

        if mask.ndim == 1:
            # 2560 is likely 80x32 or 64x40. 
            # But more importantly, we need it to match the image dimensions.
            # If SAM returned a flattened logit, we must reshape it.
            try:
                # Assuming typical SAM mask logit shapes, but let's be safe:
                h_feat, w_feat = 64, 40 # Adjust based on your model's feature map
                mask = mask.reshape(h_feat, w_feat)
            except ValueError:
                print(f"Critically malformed mask shape: {mask.shape}")
                return None

        if method == "crop":
            # Crop to bounding box of mask
            coords = np.where(mask > 0)
            # Check if the mask actually contains any pixels
            if len(coords[0]) == 0 or len(coords[1]) == 0:
                print("Warning: Empty mask detected. Skipping feature extraction for this frame.")
                return None
            # if len(coords[0]) == 0:
            #     # Empty mask, return global features
            #     return self.extract_image_features(image)
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Add margin
            margin = int(0.1 * max(x_max - x_min, y_max - y_min))
            h, w = image.shape[:2]
            y_min = max(0, y_min - margin)
            y_max = min(h, y_max + margin)
            x_min = max(0, x_min - margin)
            x_max = min(w, x_max + margin)
            
            cropped = image[y_min:y_max, x_min:x_max]
            return self.extract_image_features(cropped)
        
        else:  # inpaint
            # Zero out background
            image_masked = image.copy()
            image_masked[mask == 0] = 0
            return self.extract_image_features(image_masked)
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """
        Extract CLIP features from text.
        
        Args:
            text: Text query
        
        Returns:
            Feature vector (feature_dim,) normalized to unit length
        """
        text_tokens = self.tokenizer(text).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features[0].cpu().numpy()


def load_ply_point_cloud(ply_path: str) -> np.ndarray:
    """
    Load point cloud from PLY file.
    
    Args:
        ply_path: Path to PLY file
    
    Returns:
        (N, 3) array of 3D points
    """
    mesh = trimesh.load(ply_path)
    return mesh.vertices

def load_poses(poses_path: str) -> Dict[str, np.ndarray]:
    """
    Load camera poses from JSON.
    
    Args:
        poses_path: Path to poses.json
    
    Returns:
        Dict mapping frame names to (4, 4) pose matrices
    """
    with open(poses_path, 'r') as f:
        poses_data = json.load(f)
    
    poses = {}
    for frame_name, pose_list in poses_data.items():
        poses[frame_name] = np.array(pose_list, dtype=np.float32)
    
    return poses


def load_intrinsics(intrinsics_path: str) -> np.ndarray:
    """
    Load camera intrinsics from JSON.
    
    Args:
        intrinsics_path: Path to intrinsics.json
    
    Returns:
        (3, 3) intrinsic matrix
    """
    with open(intrinsics_path, 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict):
        # Format: {"fx": ..., "fy": ..., "cx": ..., "cy": ...}
        return np.array(data.get('camera_matrix'))
    else:
        # Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        return np.array(data, dtype=np.float32)


def backproject_pixels_to_rays(
    pixel_coords: np.ndarray,
    intrinsics: np.ndarray
) -> np.ndarray:
    """Backproject 2D pixels to 3D rays in camera space."""
    ones = np.ones((pixel_coords.shape[0], 1))
    pixel_homogeneous = np.hstack([pixel_coords, ones])
    
    intrinsics_inv = np.linalg.inv(intrinsics)
    rays_camera = (intrinsics_inv @ pixel_homogeneous.T).T
    
    ray_norms = np.linalg.norm(rays_camera, axis=1, keepdims=True)
    rays_normalized = rays_camera / ray_norms
    
    return rays_normalized


def transform_rays_to_world(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    pose: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform rays from camera to world space."""
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    
    # ray_origins_world = (rotation @ ray_origins.T).T + translation
    # ray_directions_world = (rotation @ ray_directions.T).T

    ray_origins_world = ray_origins @ rotation.T + translation
    ray_directions_world = ray_directions @ rotation.T
    
    direction_norms = np.linalg.norm(ray_directions_world, axis=1, keepdims=True)
    ray_directions_world = ray_directions_world / direction_norms
    
    return ray_origins_world, ray_directions_world


def distance_point_to_ray(
    points: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray
) -> np.ndarray:
    """Compute perpendicular distance from points to ray."""
    ov = points - ray_origin
    projection = np.dot(ov, ray_direction)
    closest_on_ray = ray_origin + projection[:, np.newaxis] * ray_direction
    distances = np.linalg.norm(points - closest_on_ray, axis=1)
    
    return distances

def compute_obb(
    points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute oriented bounding box using PCA (robust version).

    Args:
        points: (N, 3) array of 3D points

    Returns:
        center: (3,) center of OBB
        extent: (3,) half-lengths along principal axes
        rotation: (3, 3) rotation matrix (columns = axes)
    """
    assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"

    if len(points) < 4:
        # fallback (degenerate case)
        center = points.median(axis=0)
        extent = np.ones(3) * 1e-3
        rotation = np.eye(3)
        return center, extent, rotation

    # -----------------------------
    # 1. Center the points
    # -----------------------------
    center = np.median(points, axis=0)
    centered = points - center

    # -----------------------------
    # 2. PCA via SVD (stable)
    # -----------------------------
    # U: N×3, S: singular values, Vt: 3×3
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    # principal axes
    rotation = Vt.T  # columns = principal directions

    # -----------------------------
    # 3. Ensure right-handed system
    # -----------------------------
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1

    # -----------------------------
    # 4. Project points
    # -----------------------------
    projected = centered @ rotation

    # -----------------------------
    # 5. Compute bounds
    # -----------------------------
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)

    # center in local frame
    local_center = (mins + maxs) / 2.0

    # half-extents
    extent = (maxs - mins) / 2.0

    # -----------------------------
    # 6. Transform center back
    # -----------------------------
    # center = center + rotation @ local_center
    center = center + local_center @ rotation.T

    return center, extent, rotation

# def compute_refined_obb(points):
#     """
#     Computes OBB by aligning with the dominant plane of the cluster.
#     """
#     # 1. RANSAC to find the supporting plane (e.g., the PC back panel)
#     # We assume the plane is roughly perpendicular to one of the axes
#     ransac = RANSACRegressor(residual_threshold=0.01) # 1cm tolerance
#     X = points[:, :2]
#     y = points[:, 2]
#     ransac.fit(X, y)
    
#     # Get the plane normal
#     inlier_mask = ransac.inlier_mask_
#     plane_pts = points[inlier_mask]
    
#     # 2. PCA on INLIERS only to find principal directions on the plane
#     centroid = np.mean(plane_pts, axis=0)
#     centered_pts = plane_pts - centroid
#     covariance_matrix = np.cov(centered_pts.T)
#     eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
#     # Sort eigenvectors by eigenvalues (descending)
#     idx = eigenvalues.argsort()[::-1]
#     rotation = eigenvectors[:, idx]
    
#     # Ensure a right-handed coordinate system
#     if np.linalg.det(rotation) < 0:
#         rotation[:, 2] *= -1

#     # 3. Project points onto these axes to find the extent
#     local_pts = (points - centroid) @ rotation
#     min_p = np.min(local_pts, axis=0)
#     max_p = np.max(local_pts, axis=0)
    
#     # Refined Center and Extent
#     refined_center = centroid + (rotation @ ((min_p + max_p) / 2))
#     extent = (max_p - min_p) / 2  # Half-extents
    
#     return refined_center, extent, rotation

def compute_refined_obb(points):
    """
    Computes an Oriented Bounding Box using PCA, stabilized by median centering.
    """
    # 1. Use median for center to be robust against outliers/tails
    center_estimate = np.median(points, axis=0)
    centered_pts = points - center_estimate
    
    # 2. PCA to find principal axes
    # We use SVD for numerical stability on the centered coordinates
    _, _, vh = np.linalg.svd(centered_pts, full_matrices=False)
    rotation = vh.T # Columns are the principal directions

    # Ensure right-handedness
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1

    # 3. Project points into the local PCA coordinate system
    local_pts = centered_pts @ rotation
    
    # 4. Find the bounds in local space
    min_p = np.min(local_pts, axis=0)
    max_p = np.max(local_pts, axis=0)
    
    # 5. Compute final center and half-extents
    # The center is the midpoint of the local bounds, transformed back to world space
    local_center = (min_p + max_p) / 2
    refined_center = center_estimate + (rotation @ local_center)
    extent = (max_p - min_p) / 2
    
    return refined_center, extent, rotation

def align_to_world(axes: np.ndarray):
    """
    Align arbitrary axes to nearest world axes (X,Y,Z).
    axes: (3,3) with columns = directions
    """
    world = np.eye(3)

    aligned = np.zeros((3, 3))
    used = set()

    for i in range(3):
        dots = np.abs(axes[:, i] @ world)  # vectorized

        for j in np.argsort(-dots):  # try best first
            if j not in used:
                used.add(j)
                sign = np.sign(np.dot(axes[:, i], world[:, j]))
                aligned[:, j] = sign * axes[:, i]
                break

    return aligned

def compute_world_aligned_obb(points: np.ndarray):
    center = np.median(points, axis=0)
    centered = points - center

    # PCA
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh.T  # columns

    # align to world
    rotation = align_to_world(axes)

    # project
    proj = centered @ rotation

    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)

    extent = (maxs - mins) / 2

    # ✅ FIXED (consistent)
    refined_center = center + ((mins + maxs) / 2) @ rotation.T

    return refined_center, extent, rotation

class SemanticLifter:
    """Main pipeline for lifting semantic features to 3D."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        
        self.detector = GroundingDINODetector(config)
        self.segmenter = SAM2Segmenter(config)
        self.feature_extractor = CLIPFeatureExtractor(config)

    def lift_query(
        self,
        images,
        poses,
        intrinsics,
        point_cloud,
        text_query,
        depth_maps=None
    ):
        import numpy as np
        import cv2
        import os
        from sklearn.neighbors import NearestNeighbors

        if not images or not poses or depth_maps is None:
            return None

        all_lifted_points = []

        # -----------------------------
        # helper: mask → 3D
        # -----------------------------
        def lift_mask_to_3d(mask, depth_map, intrinsics, pose):
            ys, xs = np.where(mask > 0)

            if len(xs) < 10:
                return None

            # subsample
            if len(xs) > 5000:
                idx = np.random.choice(len(xs), 5000, replace=False)
                xs = xs[idx]
                ys = ys[idx]

            depths = depth_map[ys, xs]

            dmin, dmax = depths.min(), depths.max()
            if dmax - dmin < 1e-6:
                return None

            depths = (depths - dmin) / (dmax - dmin + 1e-8)
            depths = depths * 2.0

            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            X = (xs - cx) * depths / fx
            Y = (ys - cy) * depths / fy
            Z = depths

            pts_cam = np.stack([X, Y, Z], axis=1)
            pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])

            pts_world = (pose @ pts_cam_h.T).T[:, :3]
            return pts_world

        # -----------------------------
        # main loop
        # -----------------------------
        for frame_name, image in images.items():

            depth_map = depth_maps.get(frame_name)
            if frame_name not in poses or depth_map is None:
                continue

            pose = np.array(poses[frame_name])
            H, W = image.shape[:2]

            detections = self.detector.detect(image, text_query)
            if not detections:
                continue

            # match depth resolution
            if depth_map.shape != (H, W):
                depth_map = cv2.resize(depth_map, (W, H))

            for bbox in detections:
                mask = self.segmenter.segment(image, bbox)

                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()

                mask = mask.squeeze().astype(np.float32)
                mask_full = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)

                # normalize + threshold
                mask_norm = (mask_full - mask_full.min()) / (mask_full.max() - mask_full.min() + 1e-8)
                mask_bin = (mask_norm > 0.2).astype(np.uint8)

                if np.sum(mask_bin) < 10:
                    mask_bin = (mask_norm > 0.05).astype(np.uint8)

                pts_world = lift_mask_to_3d(mask_bin, depth_map, intrinsics, pose)

                if pts_world is not None:
                    all_lifted_points.append(pts_world)

        if len(all_lifted_points) == 0:
            print("[FAIL] no lifted points")
            return None

        lifted_points = np.vstack(all_lifted_points)
        print("Lifted points:", len(lifted_points))

        # -----------------------------
        # coarse NN matching (global)
        # -----------------------------
        nbrs_full = NearestNeighbors(n_neighbors=1).fit(point_cloud)
        _, indices_full = nbrs_full.kneighbors(lifted_points)

        coarse_points = point_cloud[indices_full[:, 0]]

        # 🔥 CRITICAL: refined center from geometry
        coarse_center = np.median(coarse_points, axis=0)

        # -----------------------------
        # spatial gating using refined center
        # -----------------------------
        pc_dists = np.linalg.norm(point_cloud - coarse_center, axis=1)

        upper = np.percentile(pc_dists, 30)
        local_pc = point_cloud[pc_dists < upper]

        print("Local PC size:", len(local_pc))

        if len(local_pc) < 50:
            upper = np.percentile(pc_dists, 50)
            local_pc = point_cloud[pc_dists < upper]

        # -----------------------------
        # NN snapping (local)
        # -----------------------------
        nbrs = NearestNeighbors(n_neighbors=1).fit(local_pc)
        distances, indices = nbrs.kneighbors(lifted_points)

        valid = distances[:, 0] < 0.03

        if np.sum(valid) < 20:
            matched_points = local_pc[indices[:, 0]]
        else:
            matched_points = local_pc[indices[valid, 0]]

        print("Matched points:", len(matched_points))

        if len(matched_points) < 20:
            return None

        # -----------------------------
        # remove planar background
        # -----------------------------
        center = np.median(matched_points, axis=0)
        centered = matched_points - center

        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        normal = eigvecs[:, 0]
        dist_plane = np.abs(centered @ normal)

        keep_plane = dist_plane < np.percentile(dist_plane, 50)
        plane_filtered = matched_points[keep_plane]

        if len(plane_filtered) < 20:
            plane_filtered = matched_points

        print("After plane filter:", len(plane_filtered))

        # -----------------------------
        # final tightening
        # -----------------------------
        center = np.median(plane_filtered, axis=0)
        dists = np.linalg.norm(plane_filtered - center, axis=1)

        keep = dists < np.percentile(dists, 50)
        final_points = plane_filtered[keep]

        if len(final_points) < 20:
            final_points = plane_filtered

        # -----------------------------
        # OBB
        # -----------------------------
        center, extent, rotation = compute_refined_obb(final_points)

        # 🔥 OPTIONAL: use coarse_center for more stable localization
        center = coarse_center

        return {
            "entity": text_query,
            "obb": {
                "center": center.tolist(),
                "extent": extent.tolist(),
                "rotation": rotation.tolist()
            }
        }

if __name__ == "__main__":
    print("Semantic Lifting Pipeline")
