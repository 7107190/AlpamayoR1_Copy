# SmolVLA for Autonomous Driving - Project Context

## 연구 목표
SmolVLA (450M VLA 모델)를 로보틱스에서 자율주행 Field Agent로 이식하여, 자연어 명령 기반 실시간 궤적 생성이 가능한지 검증한다.

## 아키텍처: 비동기 Commander-Agent

### Commander (대형 VLM, 0.5-1Hz)
- 입력: 저해상도 광각 이미지, 네비게이션 목표, Agent 행동 이력
- 출력: 자연어 명령 (예: "좌측으로 트럭 추월")
- 역할: 전략적 판단, 상황 평가, 예외 처리

### Field Agent (SmolVLA, 10-20Hz)
- 입력: 고해상도 전방 카메라 이미지 + Commander 텍스트 명령 + 현재 ego state (v, θ)
- 출력: Action tokens [acceleration, curvature] via flow matching
- 역할: 전술적 실행, 실시간 제어
- 핵심: Agent는 "reflexive reasoning" (latent space 내 암묵적 추론) 수행. 명시적 텍스트 reasoning은 200ms 지연으로 실시간 제어에 부적합.

## 모델 선택: SmolVLA
- 소스: https://github.com/huggingface/lerobot
- 모델: https://huggingface.co/lerobot/smolvla_base
- 450M 파라미터 (SmolVLM-2 backbone 16 layers + 100M Action Expert)
- Flow matching으로 action chunk 생성 (50 actions/inference)
- 비동기 추론: action 실행과 예측 분리
- 단일 GPU 학습/배포 가능

### SmolVLA 수정 사항
- Action Expert 출력 차원 변경: robot joints → (acceleration, curvature) × 2
- Euler integration layer 추가 (미분 가능) → 궤적 변환
- 입력: front_wide_120fov 카메라 + 텍스트 명령 + ego state

## Action Space: Unicycle Dynamics (AlpamayoR1 방식)

제어 입력: a = {(aᵢ, κᵢ)}₆₄ᵢ₌₁
- aᵢ = acceleration (가속도)
- κᵢ = curvature (곡률)

Euler 이산화로 궤적 변환:
```
x_{i+1} = x_i + v_i * cos(θ_i) * ΔT
y_{i+1} = y_i + v_i * sin(θ_i) * ΔT
θ_{i+1} = θ_i + v_i * κ_i * ΔT
v_{i+1} = v_i + a_i * ΔT
```
ΔT = 0.1s

### 장점
- Vehicle-agnostic (steering/throttle는 차량마다 다름)
- AlpamayoR1 기준 comfort metric 44% → 97% 개선
- 센서 노이즈에 robust
- 하류 MPC 컨트롤러가 실제 차량 명령으로 변환

### Ground Truth 생성
- Least-squares + Tikhonov regularization으로 궤적 waypoints에서 (a, κ) 도출
- AlpamayoR1 코드의 action_space/ 모듈 재사용

## 데이터셋: NVIDIA PhysicalAI-Autonomous-Vehicles

- HuggingFace: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
- 개발킷: https://github.com/NVlabs/physical_ai_av
- 1,727시간, ~228,000 클립 × 20초
- 25개국, 2,500+ 도시
- 7 카메라, LiDAR, Radar, Ego Motion
- **우리가 사용할 것**: front_wide_120fov 카메라 + ego motion + calibration만
- 라이선스: NVIDIA technology를 활용한 AV 개발 목적, 비상업적 연구 가능

### 데이터 구조
```
camera/camera_front_wide_120fov/
├── camera_front_wide_120fov.chunk_0000.zip  (약 100클립)
├── camera_front_wide_120fov.chunk_0001.zip
└── ...
calibration/  (ego motion 포함)
metadata/     (클립 필터링용)
```

### 다운로드 방법
```python
from huggingface_hub import login, snapshot_download

login(token="hf_YOUR_TOKEN")

# 메타데이터 먼저
snapshot_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
    repo_type="dataset",
    allow_patterns="metadata/*",
    local_dir="./nvidia_av_data"
)

# front_wide 카메라 첫 청크 (약 100클립)
snapshot_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
    repo_type="dataset",
    allow_patterns="camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip",
    local_dir="./nvidia_av_data"
)

# calibration + ego motion
snapshot_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
    repo_type="dataset",
    allow_patterns="calibration/*",
    local_dir="./nvidia_av_data"
)
```

### 데이터셋 선택 이유 (NAVSIM 대신 NVIDIA)
AlpamayoR1이 NVIDIA 데이터셋으로 학습되었기 때문에, auto-labeling시 domain mismatch를 피하려면 같은 데이터셋을 써야 한다. NAVSIM(nuPlan 기반)을 쓰면 다른 카메라/화각/이미지 스타일로 reasoning 품질이 떨어질 수 있다.

## 텍스트 명령 생성: AlpamayoR1 Auto-Labeling

### 문제
NVIDIA 데이터셋에는 이미지 + ego motion만 있고, Field Agent 학습에 필요한 텍스트 명령이 없다.

### 해결: AlpamayoR1-10B를 auto-labeler로 사용
- 모델: https://huggingface.co/nvidia/Alpamayo-R1-10B
- 코드: https://github.com/NVlabs/alpamayo
- 700K CoC reasoning traces로 학습됨
- 80,000시간 multi-camera 데이터 기반

### 파이프라인
1. NVIDIA 데이터셋 이미지 + ego trajectory 로드
2. AlpamayoR1 추론 → CoC reasoning trace 생성
3. Reasoning trace → Commander 스타일 명령으로 가공
   - CoC: "앞에 공사 차량이 있고, 좌측 차선이 비어있으므로 좌측으로 추월해야 한다"
   - Commander 명령: "Overtake the truck on the left"
4. (이미지, 텍스트 명령, 궤적) 튜플로 SmolVLA 학습

### AlpamayoR1 코드에서 재사용할 것
- `action_space/` — Unicycle dynamics 구현 (궤적 ↔ (a, κ) 변환)
- `load_physical_aiavdataset.py` — NVIDIA 데이터셋 로더
- 추론 파이프라인 — reasoning trace 자동 생성

## 학습 전략

### Phase 1: Field Agent 검증 (최우선)
- SmolVLA가 AV 제어를 10Hz+에서 수행 가능한지 확인
- 단순 시나리오: 차선 유지, 직진
- 고정 텍스트 명령: "Follow the lane"
- 데이터셋: NVIDIA 데이터 첫 1-2개 청크 (100-200 클립)
- 성공 기준: 10Hz 추론 + acceptable 궤적 품질

### Phase 2: Commander 라벨 생성
- AlpamayoR1-10B로 auto-labeling 파이프라인 구축
- NVIDIA 데이터셋에 텍스트 명령 생성
- 품질 검증을 위한 human sampling

### Phase 3: Commander-Agent 통합
- 라벨된 데이터로 SmolVLA fine-tuning
- 비동기 추론 테스트 (Commander 1Hz, Agent 10Hz)

### Phase 4: Edge Cases & Safety
- 추월, 차선 변경, 장애물 대응
- Safety override 메커니즘

## LeRobot 학습 명령어
```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<our_av_dataset> \
  --batch_size=64 \
  --steps=200000
```

## 미해결 사항
- VLM backbone fine-tuning 필요성 (로보틱스 → 자율주행 domain gap)
- Commander 구현: 기존 LLM 활용 vs fine-tune
- Safety layer 설계
- 최적 chunk size (SmolVLA 기본 50, AV에서는 조정 필요할 수 있음)

## 평가
- NAVSIM 벤치마크 활용 (학습 데이터와 별개로 평가 가능)
- 143팀 결과와 비교 가능

## 참고 논문
- SmolVLA: https://arxiv.org/abs/2506.01844
- AlpamayoR1: https://arxiv.org/abs/2511.00088
