# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Multi-GPU batch inference script: runs AlpamayoR1 on all clips using all available GPUs.
# Each GPU loads its own model and processes a subset of clips in parallel.

import os
import torch
import torch.multiprocessing as mp
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 데이터 로딩을 위해 _Data 디렉토리로 이동
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "_Data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "output")

CAM_NAMES = ["Cross Left 120FOV", "Front Wide 120FOV", "Cross Right 120FOV", "Front Tele 30FOV"]


def get_clip_ids():
    egomotion_dir = os.path.join(DATA_DIR, "labels", "egomotion")
    return sorted([
        f.replace(".egomotion.parquet", "")
        for f in os.listdir(egomotion_dir)
        if f.endswith(".egomotion.parquet")
    ])


def process_clip(model, processor, clip_id, device):
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
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

    # 시각화
    cam_frames = data["image_frames"].cpu()
    n_cams = cam_frames.shape[0]

    fig = plt.figure(figsize=(4 * n_cams, 8))
    gs = fig.add_gridspec(2, n_cams, height_ratios=[1, 1.5], hspace=0.25, wspace=0.05,
                          left=0.03, right=0.97, top=0.93, bottom=0.06)

    for i in range(n_cams):
        ax = fig.add_subplot(gs[0, i])
        img = cam_frames[i, 0].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.set_title(CAM_NAMES[i] if i < len(CAM_NAMES) else f"Camera {i}", fontsize=9)
        ax.axis("off")

    ax_traj = fig.add_subplot(gs[1, :])
    ax_traj.plot(gt_xy[0], gt_xy[1], "g-o", markersize=3, label="Ground Truth")
    for i in range(pred_xy.shape[0]):
        label = f"Prediction {i}" if pred_xy.shape[0] > 1 else "Prediction"
        ax_traj.plot(pred_xy[i, 0], pred_xy[i, 1], "r-x", markersize=3, alpha=0.7, label=label)
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_title(f"Prediction vs GT | minADE: {min_ade:.2f}m | CoC: {extra['cot'][0][0][0]}", fontsize=10)
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)

    fig.suptitle(f"Clip: {clip_id}", fontsize=12)
    out_path = os.path.join(OUTPUT_DIR, f"{clip_id}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {"clip_id": clip_id, "minADE": float(min_ade)}


def worker(gpu_id, num_gpus, clip_ids):
    os.chdir(DATA_DIR)
    device = f"cuda:{gpu_id}"

    # 이 GPU가 처리할 clip 분배
    my_clips = [c for i, c in enumerate(clip_ids) if i % num_gpus == gpu_id]
    total_mine = len(my_clips)

    # 이미 완료된 clip 건너뛰기
    pending = [c for c in my_clips if not os.path.exists(os.path.join(OUTPUT_DIR, f"{c}.png"))]
    skipped = total_mine - len(pending)
    if skipped > 0:
        print(f"[GPU {gpu_id}] Skipping {skipped} already completed clips")

    if not pending:
        print(f"[GPU {gpu_id}] All {total_mine} clips already done!")
        return

    print(f"[GPU {gpu_id}] Loading model on {device}... ({len(pending)} clips to process)")

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to(device)
    processor = helper.get_processor(model.tokenizer)
    print(f"[GPU {gpu_id}] Model loaded. Starting inference...")

    results = []
    failed = []

    for idx, clip_id in enumerate(pending):
        try:
            result = process_clip(model, processor, clip_id, device)
            results.append(result)
            print(f"[GPU {gpu_id}] [{idx+1}/{len(pending)}] {clip_id} | minADE: {result['minADE']:.2f}m")
        except Exception as e:
            failed.append(clip_id)
            print(f"[GPU {gpu_id}] [{idx+1}/{len(pending)}] FAILED {clip_id}: {e}")
            plt.close("all")
            torch.cuda.empty_cache()

    # GPU별 결과 저장
    result_file = os.path.join(OUTPUT_DIR, f"results_gpu{gpu_id}.json")
    with open(result_file, "w") as f:
        json.dump({"results": results, "failed": failed}, f, indent=2)

    print(f"\n[GPU {gpu_id}] Done: {len(results)} succeeded, {len(failed)} failed")
    if results:
        ades = [r["minADE"] for r in results]
        print(f"[GPU {gpu_id}] minADE — avg: {np.mean(ades):.2f}m, median: {np.median(ades):.2f}m")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    clip_ids = get_clip_ids()
    total = len(clip_ids)

    print("=" * 60)
    print(f"Multi-GPU Batch Inference")
    print(f"  GPUs: {num_gpus}")
    print(f"  Total clips: {total}")
    print(f"  Clips per GPU: ~{total // num_gpus}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    if num_gpus == 1:
        worker(0, 1, clip_ids)
    else:
        mp.spawn(worker, args=(num_gpus, clip_ids), nprocs=num_gpus, join=True)

    # 전체 결과 합산
    all_results = []
    all_failed = []
    for gpu_id in range(num_gpus):
        result_file = os.path.join(OUTPUT_DIR, f"results_gpu{gpu_id}.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
                all_results.extend(data["results"])
                all_failed.extend(data["failed"])

    print("\n" + "=" * 60)
    print(f"TOTAL: {len(all_results)} succeeded, {len(all_failed)} failed out of {total} clips")
    if all_results:
        ades = [r["minADE"] for r in all_results]
        print(f"minADE — avg: {np.mean(ades):.2f}m, median: {np.median(ades):.2f}m, "
              f"min: {np.min(ades):.2f}m, max: {np.max(ades):.2f}m")

    # 전체 결과 저장
    with open(os.path.join(OUTPUT_DIR, "results_all.json"), "w") as f:
        json.dump({"results": all_results, "failed": all_failed}, f, indent=2)

    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
