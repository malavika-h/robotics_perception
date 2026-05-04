import json
import numpy as np
from pathlib import Path


# =========================
# Utils
# =========================

def rotmat_to_qvec(R):
    """Convert rotation matrix to COLMAP quaternion (qw, qx, qy, qz)"""
    K = np.array([
        [R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
        [R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], 0, 0],
        [R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], 0],
        [R[1,2]-R[2,1], R[2,0]-R[0,2], R[0,1]-R[1,0], R[0,0]+R[1,1]+R[2,2]]
    ])
    K = K / 3.0
    w, V = np.linalg.eigh(K)
    q = V[:, np.argmax(w)]
    return np.array([q[3], q[0], q[1], q[2]])


# =========================
# Main conversion
# =========================

def write_colmap_model(images_dir, poses_path, intrinsics_path, output_dir):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    poses = json.load(open(poses_path))
    intr = json.load(open(intrinsics_path))

    K = np.array(intr["camera_matrix"])
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    width = intr["image_width"]
    height = intr["image_height"]

    # -------------------------
    # cameras.txt
    # -------------------------
    cam_path = output_dir / "cameras.txt"
    with open(cam_path, "w") as f:
        f.write("# Camera list\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS\n")
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    print("✔ cameras.txt written")

    # -------------------------
    # images.txt
    # -------------------------
    img_path = output_dir / "images.txt"

    image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))

    with open(img_path, "w") as f:
        f.write("# Image list\n")

        for i, img_path_i in enumerate(image_files, start=1):
            stem = img_path_i.stem

            # extract frame id
            if "frame_" in stem:
                key = str(int(stem.split("_")[-1]))
            else:
                key = stem

            if key not in poses:
                continue

            pose = np.array(poses[key])

            # pose is camera-to-world → convert to world-to-camera
            c2w = pose
            w2c = np.linalg.inv(c2w)

            R = w2c[:3, :3]
            t = w2c[:3, 3]

            qvec = rotmat_to_qvec(R)

            f.write(
                f"{i} "
                f"{qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{t[0]} {t[1]} {t[2]} "
                f"1 {img_path_i.name}\n"
            )
            f.write("\n")  # empty line required

    print("✔ images.txt written")

    # -------------------------
    # points3D.txt (empty)
    # -------------------------
    pts_path = output_dir / "points3D.txt"
    with open(pts_path, "w") as f:
        f.write("# Empty point cloud\n")

    print("✔ points3D.txt (empty) written")

    print("\n🎉 COLMAP model created at:", output_dir)


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    write_colmap_model(
        images_dir=args.images,
        poses_path=args.poses,
        intrinsics_path=args.intrinsics,
        output_dir=args.output
    )