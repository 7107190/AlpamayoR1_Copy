#!/usr/bin/env python3
"""
Download NVIDIA PhysicalAI-AV dataset chunk 0 for initial testing
Downloads: front_wide_120fov camera chunk 0, calibration, metadata, ego motion labels
"""

from huggingface_hub import login, snapshot_download
import os

# Configuration
HF_TOKEN = os.environ["HF_TOKEN"]
TARGET_DIR = "/scratch2/tropity24/nvidia_av_data"
REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

# Login to HuggingFace
print("Logging in to HuggingFace...")
login(token=HF_TOKEN)

# Create target directory
os.makedirs(TARGET_DIR, exist_ok=True)
print(f"Target directory: {TARGET_DIR}")

# Download metadata first (small, needed for filtering)
print("\n[1/4] Downloading metadata...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns="metadata/*",
    local_dir=TARGET_DIR,
    max_workers=4
)
print("✓ Metadata downloaded")

# Download calibration data (chunk 0 only)
print("\n[2/4] Downloading calibration (chunk 0 only)...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=[
        "calibration/camera_intrinsics/camera_intrinsics.chunk_0000.parquet",
        "calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet"
    ],
    local_dir=TARGET_DIR,
    max_workers=4
)
print("✓ Calibration chunk 0 downloaded")

# Download ego motion labels (chunk 0 only)
print("\n[3/4] Downloading ego motion labels (chunk 0 only)...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns="labels/egomotion/*chunk_0000*",
    local_dir=TARGET_DIR,
    max_workers=4
)
print("✓ Ego motion labels chunk 0 downloaded")

# Download front_wide_120fov camera chunk 0 (~100 clips)
print("\n[4/4] Downloading camera chunk 0 (this will take a while)...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns="camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip",
    local_dir=TARGET_DIR,
    max_workers=4
)
print("✓ Camera chunk 0 downloaded")

print(f"\n✅ All files downloaded to: {TARGET_DIR}")
print("\nDirectory structure:")
os.system(f"tree -L 3 {TARGET_DIR} || find {TARGET_DIR} -type f | head -20")
