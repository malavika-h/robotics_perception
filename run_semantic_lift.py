import argparse
import json
from pathlib import Path
import numpy as np
import cv2

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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text-query", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--poses-path", required=True)
    parser.add_argument("--intrinsics-path", required=True)
    parser.add_argument("--ply-path", required=True)
    parser.add_argument("--output-json", required=True)

    args = parser.parse_args()

    print("\n=== Loading data ===")

    images = load_images(args.image_dir)
    poses = load_poses(args.poses_path, images)
    intrinsics = load_intrinsics(args.intrinsics_path)
    point_cloud = load_ply_vertices(args.ply_path)

    print("\n=== Initializing pipeline ===")

    # ---- minimal config object ----
    class Config:
        def __init__(self):
            self.device = "cuda"
            self.min_points = 30
            self.grounding_dino_config= "third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
            self.grounding_dino_checkpoint = "third_party/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
            self.sam2_checkpoint = "third_party/sam2/sam2.1_hiera_large.pt"
            self.sam2_config = "sam2.1_hiera_l"
            self.clip_model = "ViT-L-14"
            self.clip_pretrained = "openai"

            self.dino_box_threshold = 0.3
            self.dino_text_threshold = 0.25

    config = Config()

    lifter = SemanticLifter(config)

    print(f"\n=== Running query: {args.text_query} ===")

    result = lifter.lift_query(
        images=images,
        poses=poses,
        intrinsics=intrinsics,
        point_cloud=point_cloud,
        text_query=args.text_query
    )

    if result is None:
        print("❌ No result found")
        return

    # ensure output dir exists
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Saved result to {output_path}")


if __name__ == "__main__":
    main()