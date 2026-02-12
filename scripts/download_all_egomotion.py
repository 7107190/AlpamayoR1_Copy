#!/usr/bin/env python3
"""
Download ALL egomotion chunks with auto-retry on rate limit
"""

import sys
import time
from huggingface_hub import login, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
import os

# Configuration
HF_TOKEN = os.environ["HF_TOKEN"]
TARGET_DIR = "/scratch2/tropity24/nvidia_av_data"
REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
RETRY_DELAY = 300  # 5 minutes

print("=" * 60, file=sys.stderr)
print("Downloading ALL egomotion chunks", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# Login
print("\nLogging in to HuggingFace...", file=sys.stderr)
login(token=HF_TOKEN)

# Download with auto-retry
retry_count = 0
max_retries = 100  # 충분히 큰 숫자

while retry_count < max_retries:
    try:
        print(f"\n[Attempt {retry_count + 1}] Downloading egomotion chunks...", file=sys.stderr)

        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="labels/egomotion/*.zip",
            local_dir=TARGET_DIR,
            max_workers=8
        )

        # 성공하면 종료
        print("\n✅ Download completed successfully!", file=sys.stderr)
        break

    except HfHubHTTPError as e:
        if "429" in str(e):
            retry_count += 1
            print(f"\n⚠️ Rate limit hit. Waiting {RETRY_DELAY}s before retry...", file=sys.stderr)
            print(f"Retry {retry_count}/{max_retries}", file=sys.stderr)
            time.sleep(RETRY_DELAY)
        else:
            print(f"\n❌ Error: {e}", file=sys.stderr)
            raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        raise

# Count downloaded files
egomotion_dir = os.path.join(TARGET_DIR, "labels", "egomotion")
if os.path.exists(egomotion_dir):
    zip_files = [f for f in os.listdir(egomotion_dir) if f.endswith(".zip")]
    parquet_files = [f for f in os.listdir(egomotion_dir) if f.endswith(".parquet")]

    print(f"\n📊 Current status:", file=sys.stderr)
    print(f"  Zip chunks: {len(zip_files)}", file=sys.stderr)
    print(f"  Parquet files: {len(parquet_files)}", file=sys.stderr)

    if zip_files:
        size = sum(os.path.getsize(os.path.join(egomotion_dir, f)) for f in zip_files)
        print(f"  Total size: {size / 1024 / 1024 / 1024:.2f} GB", file=sys.stderr)

print("\n" + "=" * 60, file=sys.stderr)
print("Complete!", file=sys.stderr)
print("=" * 60, file=sys.stderr)
