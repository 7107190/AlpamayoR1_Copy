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

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

import torch
import numpy as np

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from transformers import BitsAndBytesConfig  # <--- 이 줄 추가

# Example clip ID
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
print(f"Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
print("Dataset loaded.")
messages = helper.create_message(data["image_frames"].flatten(0, 1))

# 4-bit 설정 (메모리 최적화 + 에러 방지)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16  # 연산은 float16으로 수행하여 충돌 방지
# )
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
# model = AlpamayoR1.from_pretrained(
#     "nvidia/Alpamayo-R1-10B",
#     quantization_config=bnb_config,
#     device_map="auto"
# )
processor = helper.get_processor(model.tokenizer)

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

model_inputs = helper.to_device(model_inputs, "cuda")

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
        max_generation_length=256,
        return_extra=True,
    )

# the size is [batch_size, num_traj_sets, num_traj_samples]
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()
print("minADE:", min_ade, "meters")
print(
    "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
    "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
    "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
)

# --- 시각화: 예측 궤적 vs GT 궤적을 이미지로 저장 ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
os.makedirs(output_dir, exist_ok=True)

# 1) 궤적 비교 플롯 (Bird's Eye View)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(gt_xy[0], gt_xy[1], "g-o", markersize=3, label="Ground Truth")
for i in range(pred_xy.shape[0]):
    label = f"Prediction {i}" if pred_xy.shape[0] > 1 else "Prediction"
    ax.plot(pred_xy[i, 0], pred_xy[i, 1], "r-x", markersize=3, alpha=0.7, label=label)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title(f"Trajectory: Prediction vs Ground Truth\nminADE: {min_ade:.2f}m | CoC: {extra['cot'][0][0][0]}")
ax.legend()
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
traj_path = os.path.join(output_dir, "trajectory_comparison.png")
fig.savefig(traj_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Trajectory plot saved: {traj_path}")

# 2) 입력 카메라 이미지 저장
from PIL import Image
cam_frames = data["image_frames"].cpu()  # (N_cameras, num_frames, 3, H, W)
n_cams = cam_frames.shape[0]
fig, axes = plt.subplots(1, n_cams, figsize=(5 * n_cams, 5))
if n_cams == 1:
    axes = [axes]
cam_names = ["Cross Left 120FOV", "Front Wide 120FOV", "Cross Right 120FOV", "Front Tele 30FOV"]
for i in range(n_cams):
    img = cam_frames[i, 0].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    axes[i].imshow(img)
    axes[i].set_title(cam_names[i] if i < len(cam_names) else f"Camera {i}")
    axes[i].axis("off")
fig.suptitle(f"Input Camera Views (clip: {clip_id})", fontsize=14)
cam_path = os.path.join(output_dir, "input_cameras.png")
fig.savefig(cam_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Camera images saved: {cam_path}")
