import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

# Ensure your python path includes the depth_anything_v2 directory
# or that you are running this from the directory containing it.
from depth_anything_v2.dpt import DepthAnythingV2

# 1. Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENCODER = 'vitl'  # Matches your research preference for accuracy
INPUT_DIR = 'input_images'
OUTPUT_DIR = 'depth_maps'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# 2. Setup
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Initializing Depth-Anything-V2-{ENCODER} on {DEVICE}...")
model = DepthAnythingV2(**model_configs[ENCODER])

# Make sure you have downloaded the V2 checkpoint into the 'checkpoints' folder
checkpoint_path = f'checkpoints/depth_anything_v2_{ENCODER}.pth'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}. Download it first!")

model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model = model.to(DEVICE).eval()

# 3. Processing Loop
image_paths = list(Path(INPUT_DIR).glob("*.jpg")) + list(Path(INPUT_DIR).glob("*.png"))

print(f"Found {len(image_paths)} images. Processing...")

for img_path in tqdm(image_paths):
    # Load
    raw_img = cv2.imread(str(img_path))
    if raw_img is None:
        continue
        
    # Infer
    # infer_image handles the preprocessing and resizing internally
    depth = model.infer_image(raw_img) 
    
    # Save as .npy
    save_path = os.path.join(OUTPUT_DIR, img_path.stem + ".npy")
    np.save(save_path, depth)

print(f"Success! Depth maps stored in {OUTPUT_DIR}")