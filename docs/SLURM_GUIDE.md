# SLURM 주요 명령어 가이드

## 📋 목차
1. [Job 제출](#job-제출)
2. [Job 상태 확인](#job-상태-확인)
3. [Job 관리](#job-관리)
4. [클러스터 정보](#클러스터-정보)
5. [자주 쓰는 조합](#자주-쓰는-조합)

---

## Job 제출

### sbatch - 배치 Job 제출
```bash
# 기본 제출
sbatch run_script.sh

# 특정 파티션 지정
sbatch -p suma_rtx4090 run_script.sh

# GPU 개수 지정
sbatch --gres=gpu:2 run_script.sh

# 시간 제한 지정
sbatch --time=12:00:00 run_script.sh  # 12시간

# 여러 옵션 조합
sbatch -p suma_a100 --gres=gpu:4 --time=24:00:00 run_script.sh
```

### srun - 인터랙티브 실행 (주의: 6시간 후 자동 종료)
```bash
# GPU 1개 할당받아 인터랙티브 세션
srun --gres=gpu:1 --pty bash

# 특정 파티션에서 GPU 2개
srun -p suma_rtx4090 --gres=gpu:2 --pty bash

# 메모리 지정
srun --gres=gpu:1 --mem=32G --pty bash
```

---

## Job 상태 확인

### squeue - Job 큐 확인
```bash
# 내 Job만 보기
squeue -u $USER
squeue -u tropity24

# 특정 파티션의 Job들
squeue -p suma_rtx4090

# 상세 정보 (GPU 포함)
squeue -o "%.10i %.9P %.20j %.8u %.2t %.10M %.5D %.4C %b"

# 실행 중인 Job만
squeue -u $USER -t RUNNING

# 대기 중인 Job만
squeue -u $USER -t PENDING
```

**Job 상태 코드**:
- `PD`: Pending (대기 중)
- `R`: Running (실행 중)
- `CG`: Completing (완료 중)
- `CD`: Completed (완료)
- `F`: Failed (실패)

### sacct - Job 히스토리 확인
```bash
# 최근 Job 확인
sacct

# 특정 Job 상세 정보
sacct -j 1018906

# 오늘 실행한 Job들
sacct -S today

# 특정 기간 (최근 7일)
sacct -S $(date -d '7 days ago' +%Y-%m-%d)

# 상세 정보 (시간, 메모리 등)
sacct -j 1018906 --format=JobID,JobName,State,Elapsed,MaxRSS,AllocCPUS
```

### scontrol - Job 상세 정보
```bash
# Job 상세 정보
scontrol show job 1018906

# 노드 정보
scontrol show node cs-gpu-01

# 파티션 정보
scontrol show partition suma_rtx4090
```

---

## Job 관리

### scancel - Job 취소
```bash
# 특정 Job 취소
scancel 1018906

# 내 모든 Job 취소
scancel -u $USER

# 특정 이름의 Job들 취소
scancel -n alpamayo_r1_inference

# 대기 중인 Job만 취소
scancel -u $USER -t PENDING
```

### scontrol - Job 수정
```bash
# Job 시간 연장
scontrol update job=1018906 TimeLimit=48:00:00

# Job 우선순위 변경 (관리자만)
scontrol update job=1018906 Priority=1000
```

---

## 클러스터 정보

### sinfo - 파티션/노드 정보
```bash
# 전체 파티션 상태
sinfo

# 특정 파티션만
sinfo -p suma_rtx4090

# 상세 정보
sinfo -o "%P %A %C %l"
# %P: 파티션, %A: 노드 상태, %C: CPU 상태, %l: 시간 제한

# 노드별 상세 정보
sinfo -N -l

# GPU 정보 포함
sinfo -o "%P %N %G %C %m"
# %G: GPU (Generic Resources)
```

**노드 상태**:
- `idle`: 사용 가능
- `mix`: 일부 사용 중
- `alloc`: 완전 사용 중
- `down`: 다운

### 사용 가능한 파티션 목록
```bash
# 현재 클러스터의 파티션들
sinfo -o "%P %G %l" | sort -u

# GPU 종류별
# - suma_rtx4090: RTX 4090
# - suma_rtx3090: RTX 3090
# - big_suma_rtx3090: RTX 3090 (큰 노드)
# - suma_a6000: A6000
# - suma_a100: A100
```

---

## 자주 쓰는 조합

### 1. 내 Job 모니터링
```bash
# 실시간 모니터링
watch -n 5 'squeue -u $USER'

# Job 로그 실시간 확인
tail -f output/slurm_JOBID.out

# Job 완료 대기
while squeue -j 1018906 | grep -q 1018906; do sleep 10; done; echo "Job done!"
```

### 2. GPU 가용성 확인
```bash
# 전체 파티션 GPU 상태
sinfo -o "%P %G %A" | grep gpu

# 사용 가능한 GPU 찾기
sinfo -p suma_rtx4090 -t idle,mix -o "%N %G %C"

# 현재 GPU 사용 현황
squeue -o "%.10i %.9P %.20j %.8u %b" | grep gpu
```

### 3. Job 제출 템플릿
```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=suma_rtx4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err

echo "Job started: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Your commands here
python train.py

echo "Job finished: $(date)"
```

### 4. 리소스 사용량 확인
```bash
# Job의 메모리/CPU 사용량
sacct -j 1018906 --format=JobID,MaxRSS,Elapsed,State

# 실행 중인 Job의 효율성
seff 1018906  # (이 명령어가 설치되어 있다면)
```

---

## 💡 팁

### 주의사항 (클러스터 규칙)
1. **srun 주의**: srun은 exit 안 하면 계속 자원 점유 → 자동으로 6시간 후 취소됨
2. **sbatch 권장**: 배치 작업은 항상 sbatch 사용
3. **download는 login 서버에서**: GPU 노드에서 download 금지
4. **데이터는 NAS에**: /scratch2에 저장 (atime 30일 관리 필요)

### 효율적인 사용
```bash
# 1. 짧은 테스트는 작은 리소스로
sbatch -p suma_rtx3090 --gres=gpu:1 --time=1:00:00 test.sh

# 2. 긴 학습은 큰 리소스 + 충분한 시간
sbatch -p suma_a100 --gres=gpu:4 --time=72:00:00 train.sh

# 3. Job 제출 전 스크립트 문법 체크
bash -n run_script.sh

# 4. 로그 파일 크기 제한
#SBATCH --output=output/slurm_%j.out
# 너무 크면 --output=/dev/null 사용
```

### 유용한 별칭 (alias)
```bash
# ~/.bashrc에 추가
alias sq='squeue -u $USER'
alias si='sinfo -p suma_rtx4090'
alias gpu='sinfo -o "%P %G %A" | grep gpu'
alias myjobs='sacct -S today'
```

---

## 📚 더 알아보기

```bash
# 각 명령어 매뉴얼
man sbatch
man squeue
man scancel
man sinfo

# SLURM 환경 변수
echo $SLURM_JOB_ID
echo $SLURM_NODELIST
echo $CUDA_VISIBLE_DEVICES
```

---

**작성일**: 2026-02-11
**클러스터**: 연세대학교 인공지능대학원 데이터센터
