#!/usr/bin/env python3
"""
Download ALL camera_intrinsics chunks (pattern matching)
"""

from huggingface_hub import login, snapshot_download
import os

# Configuration
HF_TOKEN = os.environ["HF_TOKEN"]
TARGET_DIR = "/scratch2/tropity24/nvidia_av_data"
REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

print("=" * 60)
print("Downloading ALL camera_intrinsics chunks")
print("=" * 60)

# Login
print("\nLogging in to HuggingFace...")
login(token=HF_TOKEN)

# Download all camera_intrinsics (pattern matching)
print("\nDownloading camera_intrinsics (all chunks)...")
print("This may take a while...")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns="calibration/camera_intrinsics/*.parquet",
    local_dir=TARGET_DIR,
    max_workers=8
)

# Count downloaded files
intrinsics_dir = os.path.join(TARGET_DIR, "calibration", "camera_intrinsics")
files = [f for f in os.listdir(intrinsics_dir) if f.endswith(".parquet")]
print(f"\n✅ Downloaded {len(files)} camera_intrinsics chunks")

# Show file size
size = sum(os.path.getsize(os.path.join(intrinsics_dir, f)) for f in files)
print(f"Total size: {size / 1024 / 1024:.1f} MB")

print("\n" + "=" * 60)
print("Complete!")
print("=" * 60)
