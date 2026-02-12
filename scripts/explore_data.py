#!/usr/bin/env python3
"""
Explore NVIDIA PhysicalAI-AV dataset structure
Check data format, schema, and sample values
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/scratch2/tropity24/nvidia_av_data")

print("=" * 80)
print("NVIDIA PhysicalAI-AV Dataset Exploration")
print("=" * 80)

# 1. Metadata
print("\n[1] METADATA")
print("-" * 80)

metadata_files = list((DATA_DIR / "metadata").glob("*.parquet"))
for mf in metadata_files:
    print(f"\n{mf.name}:")
    df = pd.read_parquet(mf)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample:\n{df.head(3)}")

# 2. Calibration
print("\n\n[2] CALIBRATION")
print("-" * 80)

# Camera intrinsics
intrinsics_file = DATA_DIR / "calibration/camera_intrinsics/camera_intrinsics.chunk_0000.parquet"
print(f"\nCamera Intrinsics ({intrinsics_file.name}):")
df_intrinsics = pd.read_parquet(intrinsics_file)
print(f"  Shape: {df_intrinsics.shape}")
print(f"  Columns: {list(df_intrinsics.columns)}")
print(f"  Sample:\n{df_intrinsics.head(2)}")

# Sensor extrinsics
extrinsics_file = DATA_DIR / "calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet"
print(f"\nSensor Extrinsics ({extrinsics_file.name}):")
df_extrinsics = pd.read_parquet(extrinsics_file)
print(f"  Shape: {df_extrinsics.shape}")
print(f"  Columns: {list(df_extrinsics.columns)}")
print(f"  Sample:\n{df_extrinsics.head(2)}")

# 3. Ego Motion Labels (most important!)
print("\n\n[3] EGO MOTION LABELS")
print("-" * 80)

ego_files = sorted(list((DATA_DIR / "labels/egomotion").glob("*.parquet")))
print(f"Total ego motion files: {len(ego_files)}")

# Load first ego motion file
first_ego = ego_files[0]
clip_id = first_ego.stem.replace(".egomotion", "")
print(f"\nSample clip: {clip_id}")

df_ego = pd.read_parquet(first_ego)
print(f"  Shape: {df_ego.shape}")
print(f"  Columns: {list(df_ego.columns)}")
print(f"  Dtypes:\n{df_ego.dtypes}")
print(f"\n  First 5 rows:\n{df_ego.head()}")
print(f"\n  Statistics:\n{df_ego.describe()}")

# Check timestamp range
if 'timestamp' in df_ego.columns:
    duration = (df_ego['timestamp'].max() - df_ego['timestamp'].min()) / 1e6  # microseconds to seconds
    print(f"\n  Clip duration: {duration:.2f} seconds")
    print(f"  Sample rate: {len(df_ego) / duration:.1f} Hz")

# 4. Camera timestamps
print("\n\n[4] CAMERA DATA")
print("-" * 80)

camera_dir = DATA_DIR / "camera/camera_front_wide_120fov"
video_file = camera_dir / f"{clip_id}.camera_front_wide_120fov.mp4"
timestamp_file = camera_dir / f"{clip_id}.camera_front_wide_120fov.timestamps.parquet"

print(f"Sample clip: {clip_id}")
print(f"  Video: {video_file.name} ({video_file.stat().st_size / 1024 / 1024:.1f} MB)")

if timestamp_file.exists():
    df_ts = pd.read_parquet(timestamp_file)
    print(f"  Timestamps shape: {df_ts.shape}")
    print(f"  Columns: {list(df_ts.columns)}")
    print(f"  Sample:\n{df_ts.head()}")

print("\n" + "=" * 80)
print("Exploration complete!")
print("=" * 80)
