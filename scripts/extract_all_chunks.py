#!/usr/bin/env python3
"""
Extract all downloaded zip chunks into per-chunk folders.
- labels/egomotion/chunk_0000/, chunk_0001/, ...
- camera/camera_front_wide_120fov/chunk_0000/, chunk_0001/, ...

Skips already extracted chunks. Can re-run safely.
"""

import os
import sys
import zipfile
import glob
import time

BASE_DIR = "/scratch2/tropity24/nvidia_av_data"

TARGETS = [
    {
        "name": "egomotion",
        "zip_dir": os.path.join(BASE_DIR, "labels", "egomotion"),
        "pattern": "egomotion.chunk_*.zip",
    },
    {
        "name": "camera",
        "zip_dir": os.path.join(BASE_DIR, "camera", "camera_front_wide_120fov"),
        "pattern": "camera_front_wide_120fov.chunk_*.zip",
    },
]


def extract_target(target):
    name = target["name"]
    zip_dir = target["zip_dir"]
    pattern = target["pattern"]

    zip_files = sorted(glob.glob(os.path.join(zip_dir, pattern)))
    if not zip_files:
        print(f"[{name}] No zip files found. Skipping.", file=sys.stderr)
        return

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"[{name}] Found {len(zip_files)} zip files", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    extracted = 0
    skipped = 0

    for i, zip_path in enumerate(zip_files):
        filename = os.path.basename(zip_path)
        # egomotion.chunk_0000.zip -> chunk_0000
        chunk_name = filename.replace(f"{name}.", "").replace("camera_front_wide_120fov.", "").replace(".zip", "")
        chunk_dir = os.path.join(zip_dir, chunk_name)

        # Skip if already extracted
        if os.path.exists(chunk_dir) and os.listdir(chunk_dir):
            skipped += 1
            continue

        os.makedirs(chunk_dir, exist_ok=True)

        try:
            print(f"[{name}] [{i+1}/{len(zip_files)}] Extracting {filename} -> {chunk_name}/", file=sys.stderr)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(chunk_dir)
            os.remove(zip_path)
            print(f"  🗑️ Deleted {filename}", file=sys.stderr)
            extracted += 1
        except zipfile.BadZipFile:
            print(f"  ⚠️ Bad zip file: {filename} (possibly still downloading)", file=sys.stderr)
        except Exception as e:
            print(f"  ❌ Error: {e}", file=sys.stderr)

    print(f"\n[{name}] Done! Extracted: {extracted}, Skipped (already done): {skipped}", file=sys.stderr)


if __name__ == "__main__":
    print("=" * 60, file=sys.stderr)
    print("Chunk Extractor", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for target in TARGETS:
        extract_target(target)

    print("\n✅ All done!", file=sys.stderr)
