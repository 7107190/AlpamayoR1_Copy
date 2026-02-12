#!/usr/bin/env python3
"""
Download camera + egomotion chunks one by one (interleaved).
- ZIP 유지 (삭제하지 않음) — physical_ai_av가 ZIP에서 직접 읽음
- Rate limit 시 5분 대기 후 재시도
- 이미 다운로드된 청크 건너뛰기
- 껐다 켜도 이어서 다운로드 가능
"""

import sys
import os
import time
from datetime import datetime
from huggingface_hub import hf_hub_download, login
from huggingface_hub.errors import HfHubHTTPError, EntryNotFoundError

HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
BASE_DIR = "/scratch2/tropity24/nvidia_av_data"
RETRY_DELAY = 300  # 5 minutes

MAX_CHUNKS = 3146  # egomotion 기준 총 청크 수

EGO_PATH = "labels/egomotion"
EGO_PREFIX = "egomotion"
CAM_PATH = "camera/camera_front_wide_120fov"
CAM_PREFIX = "camera_front_wide_120fov"


def ts():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def count_existing():
    ego_dir = os.path.join(BASE_DIR, EGO_PATH)
    cam_dir = os.path.join(BASE_DIR, CAM_PATH)
    ego_count = len([f for f in os.listdir(ego_dir) if f.endswith(".zip")]) if os.path.exists(ego_dir) else 0
    cam_count = len([f for f in os.listdir(cam_dir) if f.endswith(".zip")]) if os.path.exists(cam_dir) else 0
    return ego_count, cam_count


def download_one(repo_path, prefix, chunk_num):
    """Download a single chunk ZIP. Returns: 'done', 'skipped', 'not_found'"""
    chunk_str = f"chunk_{chunk_num:04d}"
    zip_name = f"{prefix}.{chunk_str}.zip"
    repo_file = f"{repo_path}/{zip_name}"
    local_path = os.path.join(BASE_DIR, repo_file)

    if os.path.exists(local_path):
        return "skipped"

    while True:
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=repo_file,
                local_dir=BASE_DIR,
            )
            return "done"
        except EntryNotFoundError:
            return "not_found"
        except HfHubHTTPError as e:
            if "429" in str(e):
                print(f"{ts()} Rate limit. Waiting {RETRY_DELAY}s...", file=sys.stderr)
                time.sleep(RETRY_DELAY)
            else:
                raise
        except Exception as e:
            print(f"{ts()} Error: {e}. Retrying in 30s...", file=sys.stderr)
            time.sleep(30)


if __name__ == "__main__":
    start_time = time.time()

    print(f"{ts()} {'='*60}", file=sys.stderr)
    print(f"{ts()} Interleaved Download: {MAX_CHUNKS} chunks (ego+cam)", file=sys.stderr)
    print(f"{ts()} {'='*60}", file=sys.stderr)

    login(token=HF_TOKEN)

    ego_existing, cam_existing = count_existing()
    print(f"{ts()} 현재 상태: ego={ego_existing}/{MAX_CHUNKS}, cam={cam_existing}/{MAX_CHUNKS}", file=sys.stderr)
    print(f"{ts()} 남은 다운로드: ego={MAX_CHUNKS - ego_existing}, cam={MAX_CHUNKS - cam_existing}", file=sys.stderr)

    ego_done = False
    cam_done = False
    total_downloaded = 0
    prev_chunk_time = time.time()

    for i in range(MAX_CHUNKS):
        chunk_start = time.time()
        elapsed = (chunk_start - start_time) / 60

        # egomotion
        ego_status = "skip"
        if not ego_done:
            result = download_one(EGO_PATH, EGO_PREFIX, i)
            if result == "not_found":
                ego_done = True
                ego_status = "end"
            else:
                ego_status = result
                if result == "done":
                    total_downloaded += 1

        # camera
        cam_status = "skip"
        if not cam_done:
            result = download_one(CAM_PATH, CAM_PREFIX, i)
            if result == "not_found":
                cam_done = True
                cam_status = "end"
            else:
                cam_status = result
                if result == "done":
                    total_downloaded += 1

        chunk_elapsed = (time.time() - chunk_start) / 60

        # 실제 다운로드가 있었을 때만 로그 출력 (skip-skip은 생략)
        if ego_status != "skipped" or cam_status != "skipped":
            print(
                f"{ts()} [{i+1}/{MAX_CHUNKS}] chunk_{i:04d} | "
                f"ego={ego_status}, cam={cam_status} | "
                f"{chunk_elapsed:.1f}분 | "
                f"총 {total_downloaded}개 다운 | {elapsed:.1f}분 경과",
                file=sys.stderr
            )

        if ego_done and cam_done:
            break

    elapsed = (time.time() - start_time) / 60
    ego_final, cam_final = count_existing()
    print(f"{ts()} {'='*60}", file=sys.stderr)
    print(f"{ts()} COMPLETE | ego={ego_final}, cam={cam_final} | {elapsed:.1f}분 소요", file=sys.stderr)
    print(f"{ts()} {'='*60}", file=sys.stderr)
