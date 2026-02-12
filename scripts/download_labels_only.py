#!/usr/bin/env python3
"""
Download only ego motion labels for chunk 0
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

# Download ego motion labels (chunk 0 only)
print("\nDownloading ego motion labels (chunk 0 only)...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns="labels/egomotion/*chunk_0000*",
    local_dir=TARGET_DIR,
    max_workers=4
)
print("✓ Ego motion labels chunk 0 downloaded")

print(f"\n✅ Labels downloaded to: {TARGET_DIR}/labels/egomotion/")
os.system(f"find {TARGET_DIR}/labels -type f")
