# AlpamayoR1 Inference Input 구조

## 타임라인 (20초 클립 기준)

```
|━━━━━━━━━━━━━━━━━━━━ 20초 비디오 ━━━━━━━━━━━━━━━━━━━━|
0s    3.5s    5.1s(t0)                      11.5s     20s
       │      ↓                                 │
       └──────┴─────────────────────────────────┘
      History  ↑                              Future
              Input                          Prediction
```

## Input Components

### 1. 카메라 이미지 (4 frames)
```python
t0 = 5.1초 기준

Frame timestamps:
  - t0 - 0.3s = 4.8초
  - t0 - 0.2s = 4.9초
  - t0 - 0.1s = 5.0초
  - t0       = 5.1초

MP4 파일에서 해당 타임스탬프의 프레임을 추출:
  frames = camera.decode_images_from_timestamps([4.8s, 4.9s, 5.0s, 5.1s])
  → Shape: (4, 1080, 1920, 3)
```

**4개 카메라 × 4 frames = 총 16개 이미지 입력**

### 2. Ego History (과거 궤적)
```python
num_history_steps = 16
time_step = 0.1s

History timestamps: [t0-1.5s, t0-1.4s, ..., t0-0.1s, t0]
                  = [3.6s, 3.7s, ..., 5.0s, 5.1s]
                    (16 points, 1.6초 구간)

각 point마다:
  - position: (x, y, z)
  - orientation: (qx, qy, qz, qw)
  - velocity: (vx, vy, vz)
  - acceleration: (ax, ay, az)
  - curvature

Shape: (1, 1, 16, 3) for xyz
       (1, 1, 16, 3, 3) for rotation matrix
```

### 3. Ego Future (예측 대상)
```python
num_future_steps = 64
time_step = 0.1s

Future timestamps: [t0+0.1s, t0+0.2s, ..., t0+6.4s]
                 = [5.2s, 5.3s, ..., 11.5s]
                   (64 points, 6.4초 구간)

Ground Truth로 사용:
  - position: (x, y, z)
  - rotation: rotation matrix

Shape: (1, 1, 64, 3) for xyz
```

## 왜 t0 = 5.1초인가?

```
t0 = 5.1초 선택 이유:

1. History 확보: 1.6초 필요 → 최소 t0 > 1.6초
2. Future 확보: 6.4초 필요 → t0 + 6.4s < 20s → t0 < 13.6초
3. 5.1초: 충분한 context + 충분한 prediction 구간

따라서 유효 구간: 1.6초 < t0 < 13.6초
```

## 전체 Input 정리

```python
model_inputs = {
    # 1. 비전 입력
    "image_frames": torch.Tensor,  # (4 cameras, 4 frames, 3, H, W)
    "camera_indices": [0, 1, 2, 6],  # 카메라 ID
    "relative_timestamps": torch.Tensor,  # 각 프레임의 상대 시간

    # 2. 궤적 입력 (History)
    "ego_history_xyz": torch.Tensor,  # (1, 1, 16, 3)
    "ego_history_rot": torch.Tensor,  # (1, 1, 16, 3, 3)

    # 3. 메타데이터
    "t0_us": 5_100_000,  # 기준 시점
    "clip_id": "...",
}

# Ground Truth (학습/평가용)
gt = {
    "ego_future_xyz": torch.Tensor,  # (1, 1, 64, 3)
    "ego_future_rot": torch.Tensor,  # (1, 1, 64, 3, 3)
}
```

## 모델 출력

```python
pred_xyz, pred_rot, extra = model.sample_trajectories(...)

# pred_xyz: (1, 1, num_samples, 64, 3) - 예측 궤적
# extra['cot']: Chain of Thought reasoning trace
```

## 데이터 흐름

```
MP4 Video (20s)
    ↓ decode_images_from_timestamps([4.8s, 4.9s, 5.0s, 5.1s])
4 Image Frames ──────────────┐
                             ├→ AlpamayoR1 → pred_xyz (64 points)
Ego Motion Parquet          │              → CoC reasoning
    ↓ interpolate           │
16 History Points ───────────┘
```

## 예제

```python
# 클립 로드
data = load_physical_aiavdataset(
    clip_id="01d3588e-bca7-4a18-8e74-c6cfe9e996db",
    t0_us=5_100_000,  # 5.1초
    num_frames=4,     # 이미지 4장
    num_history_steps=16,  # 과거 1.6초
    num_future_steps=64,   # 미래 6.4초
)

print(f"Images: {data['image_frames'].shape}")
# → Images: torch.Size([4, 4, 3, 1080, 1920])
#           (4 cameras, 4 frames, RGB, H, W)

print(f"History: {data['ego_history_xyz'].shape}")
# → History: torch.Size([1, 1, 16, 3])

print(f"Future GT: {data['ego_future_xyz'].shape}")
# → Future GT: torch.Size([1, 1, 64, 3])
```

## 핵심 요약

**Input**:
- 📷 **4개 카메라 × 4 프레임** = 16장 이미지 (t0 기준 과거 0.3초~t0)
- 📍 **16 points 과거 궤적** (t0 기준 과거 1.6초~t0)

**Output**:
- 🎯 **64 points 미래 궤적** (t0 기준 미래 0.1초~6.4초)
- 💭 **CoC reasoning trace** (자연어)

**시간 구조**:
```
Past ─────── Present ─────── Future
[====16====][4 images][=====64=====]
1.6s history   t0      6.4s prediction
(ego motion) (vision)   (output)
```
