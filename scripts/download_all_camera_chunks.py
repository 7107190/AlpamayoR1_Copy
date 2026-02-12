#!/usr/bin/env python3
"""
Download ALL camera_front_wide_120fov chunks
"""

import sys
sys.stdout = sys.stderr  # Force immediate output

from huggingface_hub import login, snapshot_download
import os

# Configuration
HF_TOKEN = os.environ["HF_TOKEN"]
TARGET_DIR = "/scratch2/tropity24/nvidia_av_data"
REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

print("=" * 60)
print("Downloading ALL camera_front_wide_120fov chunks")
print("=" * 60)

# Login
print("\nLogging in to HuggingFace...")
login(token=HF_TOKEN)

# Download all camera chunks (pattern matching)
print("\nDownloading camera_front_wide_120fov (all chunks)...")
print("This will take a while (estimated ~95GB total)...")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns="camera/camera_front_wide_120fov/*.zip",
    local_dir=TARGET_DIR,
    max_workers=8
)

# Count downloaded files
camera_dir = os.path.join(TARGET_DIR, "camera", "camera_front_wide_120fov")
files = [f for f in os.listdir(camera_dir) if f.endswith(".zip")]
print(f"\n✅ Downloaded {len(files)} camera chunks")

# Show file size
size = sum(os.path.getsize(os.path.join(camera_dir, f)) for f in files)
print(f"Total size: {size / 1024 / 1024 / 1024:.1f} GB")

print("\n" + "=" * 60)
print("Complete!")
print("=" * 60)
