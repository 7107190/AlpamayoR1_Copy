"""
chunk_0000의 첫 번째 클립 1개만 테스트
문제 디버깅용
"""

import os
import sys
import torch
import time
import zipfile
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..", "..")
DATA_DIR = os.path.join(PROJECT_DIR, "_Data")

print(f"DATA_DIR: {os.path.realpath(DATA_DIR)}")
sys.stdout.flush()

# 1. ZIP에서 첫 클립 ID 가져오기
ego_zip = os.path.join(DATA_DIR, "labels", "egomotion", "egomotion.chunk_0000.zip")
print(f"Egomotion ZIP exists: {os.path.exists(ego_zip)}")

with zipfile.ZipFile(ego_zip, 'r') as zf:
    parquets = [n for n in zf.namelist() if n.endswith(".egomotion.parquet")]
    clip_id = os.path.basename(parquets[0]).replace(".egomotion.parquet", "")
    print(f"Test clip: {clip_id}")
sys.stdout.flush()

# 2. physical_ai_av 로드 테스트
print("\nStep 1: Loading physical_ai_av...")
sys.stdout.flush()
t0 = time.time()

import physical_ai_av
avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
print(f"  Done ({time.time()-t0:.1f}s)")
sys.stdout.flush()

# 3. 데이터 로딩 테스트
print("\nStep 2: Loading clip data (maybe_stream=True)...")
sys.stdout.flush()
t0 = time.time()

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000, avdi=avdi)
print(f"  Done ({time.time()-t0:.1f}s)")
print(f"  image_frames shape: {data['image_frames'].shape}")
sys.stdout.flush()

# 4. 모델 로딩
print("\nStep 3: Loading model...")
sys.stdout.flush()
t0 = time.time()

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda:0")
processor = helper.get_processor(model.tokenizer)
print(f"  Done ({time.time()-t0:.1f}s)")
sys.stdout.flush()

# 5. 추론 테스트
print("\nStep 4: Running inference...")
sys.stdout.flush()
t0 = time.time()

messages = helper.create_message(data["image_frames"].flatten(0, 1))
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    continue_final_message=True, return_dict=True, return_tensors="pt",
)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
model_inputs = helper.to_device(model_inputs, "cuda:0")

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs, top_p=0.98, temperature=0.6,
        num_traj_samples=1, max_generation_length=256, return_extra=True,
    )
print(f"  Done ({time.time()-t0:.1f}s)")
sys.stdout.flush()

# 6. 결과
import numpy as np
gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()

reasoning = extra['cot'][0][0][0] if 'cot' in extra else ""

print(f"\n{'='*60}")
print(f"SUCCESS!")
print(f"  Clip: {clip_id}")
print(f"  minADE: {min_ade:.2f}m")
print(f"  Reasoning: {reasoning[:200]}...")
print(f"{'='*60}")
sys.stdout.flush()
