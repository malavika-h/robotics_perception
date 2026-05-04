import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import os

# ---- import your pipeline ----
# make sure this matches your file name
from lifting_3d_pipeline import SemanticLifter

# -------------------------------
# Utils
# -------------------------------load

def load_images(image_dir):
    image_dir = Path(image_dir)
    images = {}

    for p in sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[p.name] = img

    print(f"Loaded {len(images)} images")
    return images

def load_depth_maps(depth_dir: str) -> dict:
    depth_maps = {}

    for f in os.listdir(depth_dir):
        if f.endswith(".npy"):
            name = os.path.splitext(f)[0]          # frame_000123_depth
            name = name.replace("_depth", ".png")      # frame_000123

            depth_maps[name] = np.load(os.path.join(depth_dir, f))
    print(f"Loaded {len(depth_maps)} depth maps")
    return depth_maps

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_intrinsics(path):
    data = load_json(path)
    K = np.array(data["camera_matrix"], dtype=np.float32)

    print("Camera intrinsics:\n", K)
    return K


def load_poses(path, images):
    raw = load_json(path)

    poses = {}

    for k, v in raw.items():
        # convert "319" → "frame_000319.png"
        fname = f"frame_{int(k):06d}.png"

        if fname in images:
            poses[fname] = np.array(v, dtype=np.float32)

    print(f"Matched {len(poses)} poses with images")
    return poses

import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# -----------------------------
# LOAD MODELS (assumed available)
# -----------------------------
# You should already have these wrappers from your pipeline
# detector.detect(image, text)
# segmenter.segment(image, bbox)
# depth_model(image)

# -----------------------------
# OBB via PCA
# -----------------------------
def compute_obb(points):
    center = np.mean(points, axis=0)

    centered = points - center
    cov = np.cov(centered.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    # sort descending
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # project points
    proj = centered @ eigvecs

    min_p = proj.min(axis=0)
    max_p = proj.max(axis=0)

    extent = max_p - min_p

    return center, extent, eigvecs


# -----------------------------
# LIFT MASK → 3D
# -----------------------------
def lift_mask_to_3d(mask, depth_map, intrinsics, pose):
    ys, xs = np.where(mask > 0)

    if len(xs) < 20:
        return None

    # subsample for stability
    if len(xs) > 5000:
        idx = np.random.choice(len(xs), 5000, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    z = depth_map[ys, xs]

    # normalize depth (Depth Anything is relative)
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    z = z * 2.0

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    Z = z

    pts_cam = np.stack([X, Y, Z], axis=1)

    # to world
    pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
    pts_world = (pose @ pts_cam_h.T).T[:, :3]

    return pts_world


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def run_single_image_obb(
    image_path,
    intrinsics,
    pose,
    output_path,
    queries
):
    image = cv2.imread(image_path)
    H, W = image.shape[:2]

    # -----------------------------
    # DEPTH
    # -----------------------------
    with torch.no_grad():
        depth = depth_model(image)

    depth = F.interpolate(
        depth[None],
        (H, W),
        mode='bilinear',
        align_corners=False
    )[0, 0].cpu().numpy()

    results = []

    for query in queries:
        print(f"\nProcessing: {query}")

        detections = detector.detect(image, query)

        if not detections:
            print("No detection")
            continue

        # take best bbox
        bbox = detections[0]

        mask = segmenter.segment(image, bbox)

        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()

        mask = mask.squeeze().astype(np.float32)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)

        # normalize mask
        mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask_bin = (mask_norm > 0.2).astype(np.uint8)

        if np.sum(mask_bin) < 20:
            mask_bin = (mask_norm > 0.05).astype(np.uint8)

        # -----------------------------
        # LIFT
        # -----------------------------
        pts = lift_mask_to_3d(mask_bin, depth, intrinsics, pose)

        if pts is None:
            print("No valid 3D points")
            continue

        # -----------------------------
        # DEPTH CONSISTENCY FILTER
        # -----------------------------
        z_vals = pts[:, 2]
        med = np.median(z_vals)

        keep = np.abs(z_vals - med) < 0.1
        pts = pts[keep]

        if len(pts) < 20:
            print("Too few points after filtering")
            continue

        # -----------------------------
        # OBB
        # -----------------------------
        center, extent, rotation = compute_obb(pts)

        results.append({
            "entity": query,
            "obb": {
                "center": center.tolist(),
                "extent": extent.tolist(),
                "rotation": rotation.tolist()
            }
        })

    # -----------------------------
    # SAVE
    # -----------------------------
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to:", output_path)

def load_ply_vertices(path, limit=300_000):
    from plyfile import PlyData

    ply = PlyData.read(path)
    v = ply["vertex"]

    pts = np.vstack([v["x"], v["y"], v["z"]]).T
    if len(pts) > limit:
        idx = np.random.choice(len(pts), limit, replace=False)
        pts = pts[idx]

    print(f"Loaded {len(pts)} 3D points")
    return pts


# -------------------------------
# Main
# -------------------------------

# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--text-query", required=True)
#     parser.add_argument("--image-dir", required=True)
#     parser.add_argument("--poses-path", required=True)
#     parser.add_argument("--intrinsics-path", required=True)
#     parser.add_argument("--ply-path", required=True)
#     parser.add_argument("--depth-path", required=True)
#     parser.add_argument("--output-json", required=True)

#     args = parser.parse_args()

#     print("\n=== Loading data ===")

#     images = load_images(args.image_dir)
#     poses = load_poses(args.poses_path, images)
#     intrinsics = load_intrinsics(args.intrinsics_path)
#     point_cloud = load_ply_vertices(args.ply_path)
#     depth_maps = load_depth_maps(args.depth_path)

#     print("\n=== Initializing pipeline ===")

#     # ---- minimal config object ----
#     class Config:
#         def __init__(self):
#             self.device = "cuda"
#             self.min_points = 30
#             self.grounding_dino_config= "third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
#             self.grounding_dino_checkpoint = "third_party/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
#             self.sam2_checkpoint = "third_party/sam2/sam2.1_hiera_large.pt"
#             self.sam2_config = "sam2.1_hiera_l"
#             self.clip_model = "ViT-L-14"
#             self.clip_pretrained = "openai"

#             self.dino_box_threshold = 0.3
#             self.dino_text_threshold = 0.25

#     config = Config()

#     lifter = SemanticLifter(config)

#     print(f"\n=== Running query: {args.text_query} ===")

#     result = lifter.lift_query(
#         images=images,
#         poses=poses,
#         intrinsics=intrinsics,
#         point_cloud=point_cloud,
#         text_query=args.text_query,
#         depth_maps= depth_maps
#     )

#     if result is None:
#         print("❌ No result found")
#         return

#     # ensure output dir exists
#     output_path = Path(args.output_json)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(output_path, "w") as f:
#         json.dump(result, f, indent=2)

#     print(f"\n✅ Saved result to {output_path}")


# # if __name__ == "__main__":
# #     main()

# # -----------------------------
# # USAGE
# # -----------------------------
# if __name__ == "__main__":

#     queries = [
#         "ethernet_socket",
#         "power_socket",
#         "hdmi_socket_left",
#         "usb_socket_top_right"
#     ]

#     intrinsics = np.array([
#         [fx, 0, cx],
#         [0, fy, cy],
#         [0, 0, 1]
#     ])

#     pose = np.array(...)  # load from poses.json

#     run_single_image_obb(
#         image_path="frame_000465.png",
#         intrinsics=intrinsics,
#         pose=pose,
#         output_path="output.json",
#         queries=queries
#     )

# -------------------------------
# MAIN
# -------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["single", "multi"], required=True)

    parser.add_argument("--text-query", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--poses-path", required=True)
    parser.add_argument("--intrinsics-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--frame", default=None, help="image filename to process")

    # only needed for multi mode
    parser.add_argument("--ply-path", default=None)
    parser.add_argument("--depth-path", default=None)

    args = parser.parse_args()

    print("\n=== Loading data ===")

    images = load_images(args.image_dir)
    poses = load_poses(args.poses_path, images)
    intrinsics = load_intrinsics(args.intrinsics_path)

    queries = [
        "ethernet_socket",
        "power_socket",
        "hdmi_socket_left",
        "usb_socket_at_top_right"
    ]

    # =========================================================
    # 🔥 MODE 1: SINGLE IMAGE (ASSIGNMENT)
    # =========================================================
    if args.mode == "single":

        print("\n=== Running SINGLE IMAGE mode ===")

        if args.frame is None:
            raise ValueError("single mode requires --frame")

        # point_cloud = load_ply_vertices(args.ply_path)
        depth_maps = load_depth_maps(args.depth_path)
        # get single image + pose
        frame_name = args.frame

        if frame_name not in images:
            raise ValueError(f"Image not found: {frame_name}")
        if frame_name not in poses:
            raise ValueError(f"Pose not found: {frame_name}")

        image = images[frame_name]
        pose = poses[frame_name]
        depth = depth_maps[frame_name]

        # H, W = image.shape[:2]
        # depth = F.interpolate(
        #     depth[None],
        #     (H, W),
        #     mode='bilinear',
        #     align_corners=False
        # )[0, 0].cpu().numpy()

        # same config + lifter
        class Config:
            def __init__(self):
                self.device = "cuda"
                self.grounding_dino_config = "third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
                self.grounding_dino_checkpoint = "third_party/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
                self.sam2_checkpoint = "third_party/sam2/sam2.1_hiera_large.pt"
                self.sam2_config = "sam2.1_hiera_l"
                self.clip_model = "ViT-L-14"
                self.clip_pretrained = "openai"

                self.dino_box_threshold = 0.3
                self.dino_text_threshold = 0.25

        lifter = SemanticLifter(Config())

        results = []

        for query in queries:
            print(f"\n=== Running query: {query} ===")

            result = lifter.lift_query_single(
                image=image,
                pose=pose,
                intrinsics=intrinsics,
                text_query=query,
                depth_map=depth
            )

            if result is not None:
                results.append(result)

        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)

        print("\n✅ SINGLE IMAGE DONE")
    # =========================================================
    # 🔥 MODE 2: MULTI-VIEW PIPELINE
    # =========================================================
    elif args.mode == "multi":

        print("\n=== Running MULTI-VIEW mode ===")

        if args.ply_path is None or args.depth_path is None:
            raise ValueError("multi mode requires --ply-path and --depth-path")

        point_cloud = load_ply_vertices(args.ply_path)
        depth_maps = load_depth_maps(args.depth_path)

        class Config:
            def __init__(self):
                self.device = "cuda"
                self.grounding_dino_config = "third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
                self.grounding_dino_checkpoint = "third_party/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
                self.sam2_checkpoint = "third_party/sam2/sam2.1_hiera_large.pt"
                self.sam2_config = "sam2.1_hiera_l"
                self.clip_model = "ViT-L-14"
                self.clip_pretrained = "openai"

                self.dino_box_threshold = 0.3
                self.dino_text_threshold = 0.25

        lifter = SemanticLifter(Config())

        results = []

        for query in queries:
            print(f"\n=== Running query: {query} ===")

            result = lifter.lift_query(
                images=images,
                poses=poses,
                intrinsics=intrinsics,
                point_cloud=point_cloud,
                text_query=query,
                depth_maps=depth_maps
            )

            if result is not None:
                results.append(result)

        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)

        print("\n✅ MULTI-VIEW DONE")


if __name__ == "__main__":
    main()