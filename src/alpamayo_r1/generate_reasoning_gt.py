# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate reasoning ground truth using AlpamayoR1
- Multi-GPU: 각 GPU에 청크를 분배하여 병렬 처리
- 풀 스트리밍: egomotion + camera 모두 HuggingFace에서 스트리밍
- 재개 가능: 이미 완료된 클립은 건너뜀 (JSON 파일 존재 여부)
- 진행 상황: 실시간 로그 (GPU별 + 타임스탬프)
"""

import os
import sys
import time
import json
import zipfile
from datetime import datetime

import torch
import torch.multiprocessing as mp
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..", "..")
DATA_DIR = os.path.join(PROJECT_DIR, "_Data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
REASONING_DIR = os.path.join(DATA_DIR, "labels", "reasoning")

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
MAX_CHUNKS = 3146


def ts():
    return datetime.now().strftime("[%H:%M:%S]")


def get_clip_ids_from_chunk(chunk_name):
    """egomotion ZIP에서 clip ID 추출 (로컬 → HF 다운로드 fallback)"""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    # 로컬 체크
    local_path = os.path.join(DATA_DIR, "labels", "egomotion", f"egomotion.{chunk_name}.zip")
    if os.path.exists(local_path):
        zip_path = local_path
    else:
        # HF cache로 다운로드
        try:
            zip_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=f"labels/egomotion/egomotion.{chunk_name}.zip",
            )
        except EntryNotFoundError:
            return None
        except Exception as e:
            print(f"{ts()} Warning: {chunk_name} egomotion download error: {e}")
            return None

    clip_ids = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith(".egomotion.parquet"):
                clip_id = os.path.basename(name).replace(".egomotion.parquet", "")
                clip_ids.append(clip_id)
    return sorted(clip_ids)


def process_clip(model, processor, avdi, clip_id, device):
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000, avdi=avdi)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()

    reasoning_trace = extra['cot'][0][0][0] if 'cot' in extra else ""

    return {
        "reasoning": reasoning_trace,
        "minADE": float(min_ade),
        "gt_trajectory": gt_xy.tolist(),
        "pred_trajectory": pred_xy[diff.argmin()].tolist(),
    }


def gpu_worker(rank, world_size, max_chunks, reasoning_dir):
    """각 GPU 워커: 할당된 청크를 순회하며 처리"""
    import physical_ai_av
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    import matplotlib
    matplotlib.use("Agg")

    device = f"cuda:{rank}"
    print(f"{ts()} [GPU{rank}] Loading model on {device}...")
    sys.stdout.flush()

    t0 = time.time()
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to(device)
    processor = helper.get_processor(model.tokenizer)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    print(f"{ts()} [GPU{rank}] Model loaded ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    total_succeeded = 0
    total_failed = 0
    total_skipped = 0

    # 이 워커가 담당하는 청크: rank, rank+world_size, rank+2*world_size, ...
    for chunk_idx in range(rank, max_chunks, world_size):
        chunk_name = f"chunk_{chunk_idx:04d}"

        clip_ids = get_clip_ids_from_chunk(chunk_name)
        if clip_ids is None:
            continue

        # Pending 클립 필터
        chunk_dir = os.path.join(reasoning_dir, chunk_name)
        pending = []
        for clip_id in clip_ids:
            json_path = os.path.join(chunk_dir, f"{clip_id}.reasoning.json")
            if not os.path.exists(json_path):
                pending.append(clip_id)

        if not pending:
            total_skipped += len(clip_ids)
            continue

        total_skipped += len(clip_ids) - len(pending)
        os.makedirs(chunk_dir, exist_ok=True)

        print(f"{ts()} [GPU{rank}] {chunk_name}: {len(pending)} pending / {len(clip_ids)} total")
        sys.stdout.flush()

        for i, clip_id in enumerate(pending):
            clip_start = time.time()
            try:
                result = process_clip(model, processor, avdi, clip_id, device)

                reasoning_data = {
                    "clip_id": clip_id,
                    "chunk": chunk_name,
                    **result,
                }
                json_path = os.path.join(chunk_dir, f"{clip_id}.reasoning.json")
                with open(json_path, "w") as f:
                    json.dump(reasoning_data, f, indent=2)

                total_succeeded += 1
                elapsed = time.time() - clip_start
                print(
                    f"{ts()} [GPU{rank}] {chunk_name} [{i+1}/{len(pending)}] "
                    f"{clip_id} | minADE: {result['minADE']:.2f}m | {elapsed:.1f}s"
                )
                sys.stdout.flush()

            except Exception as e:
                total_failed += 1
                print(f"{ts()} [GPU{rank}] {chunk_name} [{i+1}/{len(pending)}] FAILED {clip_id}: {e}")
                sys.stdout.flush()
                import matplotlib.pyplot as plt
                plt.close("all")
                torch.cuda.empty_cache()

    print(
        f"{ts()} [GPU{rank}] DONE | "
        f"Succeeded: {total_succeeded}, Failed: {total_failed}, Skipped: {total_skipped}"
    )
    sys.stdout.flush()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REASONING_DIR, exist_ok=True)

    world_size = torch.cuda.device_count()

    print(f"{ts()} Reasoning GT Generation (Full Streaming)")
    print(f"{ts()} GPUs: {world_size}")
    print(f"{ts()} Max chunks: {MAX_CHUNKS}")
    print(f"{ts()} Chunks per GPU: ~{MAX_CHUNKS // max(world_size, 1)}")
    sys.stdout.flush()

    if world_size > 1:
        mp.spawn(gpu_worker, args=(world_size, MAX_CHUNKS, REASONING_DIR), nprocs=world_size)
    else:
        gpu_worker(0, 1, MAX_CHUNKS, REASONING_DIR)

    # 최종 결과 집계
    done_count = 0
    chunk_count = 0
    if os.path.exists(REASONING_DIR):
        for d in sorted(os.listdir(REASONING_DIR)):
            chunk_dir = os.path.join(REASONING_DIR, d)
            if os.path.isdir(chunk_dir):
                files = [f for f in os.listdir(chunk_dir) if f.endswith(".reasoning.json")]
                if files:
                    chunk_count += 1
                    done_count += len(files)

    print(f"\n{'='*60}")
    print(f"{ts()} ALL COMPLETE")
    print(f"  Chunks with results: {chunk_count}")
    print(f"  Total reasoning files: {done_count}")
    print(f"{'='*60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
