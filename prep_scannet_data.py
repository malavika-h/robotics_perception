import os
import json
import numpy as np
import cv2
import shutil
from pathlib import Path

def generate_scannet():
    # 1. SETUP PATHS
    base_dir = Path(__file__).resolve().parents[0]
    data_dir = base_dir / "Data"
    output_dir = data_dir / "scannet_format"
    
    # Input files
    poses_path = data_dir / "poses.json"
    intrinsic_path = data_dir / "intrinsic.json"
    images_src_dir = data_dir / "images"
    
    # Create ScanNet folders
    for folder in ['color', 'pose', 'intrinsic', 'depth']:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    # 2. LOAD METADATA
    with open(intrinsic_path, 'r') as f:
        int_data = json.load(f)
    K = np.array(int_data['camera_matrix'])

    with open(poses_path, 'r') as f:
        poses_dict = json.load(f)

    # 3. PROCESS VALID FRAMES
    # We only process images that have a corresponding pose in poses.json
    frame_keys = sorted(poses_dict.keys())
    print(f"Found {len(frame_keys)} posed frames. Starting conversion...")

    for i, frame_id in enumerate(frame_keys):
        # Determine image path (handles potential naming conventions like frame_000319.png)
        img_name = f"frame_{int(frame_id):06d}.png" 
        src_img_path = images_src_dir / img_name
        
        if not src_img_path.exists():
            # Fallback check if your naming is different
            img_name = f"{frame_id}.png"
            src_img_path = images_src_dir / img_name
            if not src_img_path.exists():
                print(f"Warning: Image {frame_id} not found, skipping.")
                continue

        # --- A. Save Color Image ---
        # Standardizing to {index}.png for ScanNet compatibility
        shutil.copy(src_img_path, output_dir / 'color' / f"{i}.png")

        # --- B. Save Pose (4x4 matrix) ---
        pose_matrix = np.array(poses_dict[frame_id])
        np.savetxt(output_dir / 'pose' / f"{i}.txt", pose_matrix)

        # --- C. Save Intrinsic (3x3 matrix) ---
        np.savetxt(output_dir / 'intrinsic' / f"{i}.txt", K)

        # --- D. Depth Map Processing ---
        # Note: This script assumes you run DepthAnything if depth doesn't exist
        # If you already have depth maps, point to that directory here.
        # Otherwise, this placeholder logic reminds you to save them as 16-bit PNGs.
        placeholder_depth = np.zeros((int_data['image_height'], int_data['image_width']), dtype=np.uint16)
        cv2.imwrite(str(output_dir / 'depth' / f"{i}.png"), placeholder_depth)

    # --- E. Geometry Anchor ---
    fused_ply = base_dir / "outputs" / "fused_pointcloud.ply"
    if fused_ply.exists():
        shutil.copy(fused_ply, output_dir / "points3d.ply")
        print("Fused PLY copied as points3d.ply for initialization.")

    print(f"\nScanNet dataset ready at: {output_dir}")
    print("Next step: Run CLIP distillation on the 'color' folder.")

if __name__ == "__main__":
    generate_scannet()