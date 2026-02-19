#!/usr/bin/env python3
"""
Download all 4 cameras + egomotion chunks one by one (interleaved).
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

MAX_CHUNKS = 3146

# 다운로드 대상: 아직 완료되지 않은 카메라만
# egomotion, camera_front_wide_120fov 는 이미 3146/3146 완료
FEATURES = [
    ("camera/camera_cross_left_120fov", "camera_cross_left_120fov"),
    ("camera/camera_cross_right_120fov", "camera_cross_right_120fov"),
    ("camera/camera_front_tele_30fov", "camera_front_tele_30fov"),
]


def ts():
    return datetime.now().strftime("[%H:%M:%S]")


def count_existing(repo_path, prefix):
    feat_dir = os.path.join(BASE_DIR, repo_path)
    if not os.path.exists(feat_dir):
        return 0
    return len([f for f in os.listdir(feat_dir) if f.endswith(".zip")])


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
                print(f"{ts()} Rate limit. Waiting {RETRY_DELAY}s...", flush=True)
                time.sleep(RETRY_DELAY)
            else:
                raise
        except Exception as e:
            print(f"{ts()} Error: {e}. Retrying in 30s...", flush=True)
            time.sleep(30)


if __name__ == "__main__":
    start_time = time.time()

    print(f"{ts()} {'='*60}", flush=True)
    print(f"{ts()} Download: {MAX_CHUNKS} chunks x {len(FEATURES)} features", flush=True)
    print(f"{ts()} Features: {[p for _, p in FEATURES]}", flush=True)
    print(f"{ts()} {'='*60}", flush=True)

    login(token=HF_TOKEN)

    # 현재 상태 출력
    for repo_path, prefix in FEATURES:
        cnt = count_existing(repo_path, prefix)
        print(f"{ts()} {prefix}: {cnt}/{MAX_CHUNKS}", flush=True)

    # 500번대부터 시작 → 끝까지 → 0부터 빈 것 채우기
    START = 500
    chunk_order = list(range(START, MAX_CHUNKS)) + list(range(0, START))
    print(f"{ts()} Starting from chunk_{START:04d}", flush=True)

    feat_done = [False] * len(FEATURES)
    total_downloaded = 0

    for i in chunk_order:
        chunk_start = time.time()
        elapsed = (chunk_start - start_time) / 60
        statuses = {}

        for j, (repo_path, prefix) in enumerate(FEATURES):
            if feat_done[j]:
                statuses[prefix] = "end"
                continue

            result = download_one(repo_path, prefix, i)
            if result == "not_found":
                feat_done[j] = True
                statuses[prefix] = "end"
            else:
                statuses[prefix] = result
                if result == "done":
                    total_downloaded += 1

        chunk_elapsed = (time.time() - chunk_start) / 60

        # 실제 다운로드가 있었을 때만 로그 출력
        if any(s not in ("skipped", "end") for s in statuses.values()):
            status_str = ", ".join(f"{k.split('_')[-1]}={v}" for k, v in statuses.items())
            print(
                f"{ts()} chunk_{i:04d} | "
                f"{status_str} | "
                f"{chunk_elapsed:.1f}분 | "
                f"총 {total_downloaded}개 다운 | {elapsed:.1f}분 경과",
                flush=True,
            )

        if all(feat_done):
            break

    elapsed = (time.time() - start_time) / 60
    print(f"{ts()} {'='*60}", flush=True)
    for repo_path, prefix in FEATURES:
        cnt = count_existing(repo_path, prefix)
        print(f"{ts()} {prefix}: {cnt}", flush=True)
    print(f"{ts()} COMPLETE | {elapsed:.1f}분 소요", flush=True)
    print(f"{ts()} {'='*60}", flush=True)
