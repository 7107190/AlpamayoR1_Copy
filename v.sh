#!/bin/bash
# ./v.sh        한번 출력
# ./v.sh -w     2초마다 자동 갱신
# ./v.sh -w 5   5초 간격

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PROJECT_DIR/_Data"
REASONING_DIR="$DATA_DIR/labels/reasoning"
EGO_DIR="$DATA_DIR/labels/egomotion"
OUTPUT_DIR="$PROJECT_DIR/output"
MAX_CHUNKS=3146
TOTAL_CLIPS_EST=314600

if [ "$1" = "-w" ] || [ "$1" = "--watch" ]; then
    watch -n "${2:-2}" -c "$0"; exit 0
fi

B="\033[1m"; D="\033[2m"; G="\033[32m"; Y="\033[33m"; N="\033[0m"

# ── 데이터 수집 ──
CAMS=("camera_front_wide_120fov" "camera_cross_left_120fov" "camera_cross_right_120fov" "camera_front_tele_30fov")
CAM_LABELS=("front_wide" "cross_left" "cross_right" "front_tele")

declare -a CAM_COUNTS
for i in "${!CAMS[@]}"; do
    dir="$DATA_DIR/camera/${CAMS[$i]}"
    if [ -d "$dir" ]; then
        CAM_COUNTS[$i]=$(ls "$dir"/*.zip 2>/dev/null | wc -l)
    else
        CAM_COUNTS[$i]=0
    fi
done

EGO_COUNT=0; [ -d "$EGO_DIR" ] && EGO_COUNT=$(ls "$EGO_DIR"/*.zip 2>/dev/null | wc -l)

TOTAL_CLIPS=0; DONE_CHUNKS=0
if [ -d "$REASONING_DIR" ]; then
    for d in "$REASONING_DIR"/chunk_*; do
        [ -d "$d" ] || continue
        c=$(ls "$d"/*.reasoning.json 2>/dev/null | wc -l)
        [ "$c" -gt 0 ] && { DONE_CHUNKS=$((DONE_CHUNKS+1)); TOTAL_CLIPS=$((TOTAL_CLIPS+c)); }
    done
fi

DISK_INFO=$(df -h "$PROJECT_DIR" 2>/dev/null | tail -1 | awk '{printf "%s/%s(%s)", $3, $2, $5}')
HF_SIZE=""; [ -d "$HOME/.cache/huggingface" ] && HF_SIZE=$(du -sh "$HOME/.cache/huggingface" 2>/dev/null | cut -f1)

# ── SLURM ──
SLURM_LINE=$(squeue -u "$USER" -h 2>/dev/null | head -1)
if [ -n "$SLURM_LINE" ]; then
    JOB_ID=$(echo "$SLURM_LINE" | awk '{print $1}')
    JOB_TIME=$(echo "$SLURM_LINE" | awk '{print $6}')
    JOB_NODE=$(echo "$SLURM_LINE" | awk '{print $8}')
    SLURM_STR="Job ${B}${JOB_ID}${N} ${G}R${N} ${JOB_NODE} ${D}${JOB_TIME}${N}"
else
    SLURM_STR="${D}no jobs${N}"
fi

# ── 출력 ──
echo ""
echo -e "${D}$(date '+%Y-%m-%d %H:%M:%S')${N}  ${B}SLURM${N} ${SLURM_STR}  ${B}Disk${N} ${DISK_INFO} ${D}HF:${HF_SIZE:-n/a}${N}"
echo -e "${D}──────────────────────────────────────────────────────────────────────────────────────${N}"

# 카메라 4종 + egomotion (2줄로)
EGO_PCT=$((EGO_COUNT * 100 / MAX_CHUNKS))
printf "  ${B}DL${N} "
for i in "${!CAMS[@]}"; do
    cnt=${CAM_COUNTS[$i]}
    pct=$((cnt * 100 / MAX_CHUNKS))
    if [ "$pct" -eq 100 ]; then c="$G"; else c="$Y"; fi
    printf "${B}${CAM_LABELS[$i]}${N}:${c}%d%%${N} " "$pct"
done
if [ "$EGO_PCT" -eq 100 ]; then c="$G"; else c="$Y"; fi
printf "${B}ego${N}:${c}%d%%${N}\n" "$EGO_PCT"

# reasoning GT
INF_PCT=$((TOTAL_CLIPS * 100 / TOTAL_CLIPS_EST))
printf "  ${B}GT${N} ${G}%d${N}/${D}~%dk${N} clips (${G}%d%%${N}, %d chunks)\n" "$TOTAL_CLIPS" "$((TOTAL_CLIPS_EST/1000))" "$INF_PCT" "$DONE_CHUNKS"

# ── 로그 ──
RUNNING_JOB=$(squeue -u "$USER" -h -o "%i" 2>/dev/null | head -1)
if [ -n "$RUNNING_JOB" ]; then
    LOG_FILE="$OUTPUT_DIR/reasoning_${RUNNING_JOB}.out"

    if [ -f "$LOG_FILE" ]; then
        echo -e "${D}──────────────────────────────────────────────────────────────────────────────────────${N}"
        echo -e "  ${B}Log${N} $(tail -1 "$LOG_FILE" 2>/dev/null)"
    fi

    DL_LOG="$OUTPUT_DIR/download.log"
    if [ -f "$DL_LOG" ]; then
        DL_LINE=$(tail -1 "$DL_LOG" 2>/dev/null | grep -v "^$")
        [ -n "$DL_LINE" ] && echo -e "  ${B}DL${N}  ${DL_LINE}"
    fi

    # ── GPU ──
    NODE=$(squeue -j "$RUNNING_JOB" -h -o "%N" 2>/dev/null)
    if [ -n "$NODE" ]; then
        GPU_INFO=$(srun --jobid="$RUNNING_JOB" --overlap nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
        if [ -n "$GPU_INFO" ]; then
            echo -e "${D}──────────────────────────────────────────────────────────────────────────────────────${N}"
            echo -ne "  ${B}GPU${N} "
            first=1
            while IFS=',' read -r idx util mu mt; do
                idx=$(echo "$idx"|xargs); util=$(echo "$util"|xargs)
                mu=$(echo "$mu"|xargs); mt=$(echo "$mt"|xargs)
                mem_gb=$(awk "BEGIN{printf \"%.0f\", $mu/1024}")
                tot_gb=$(awk "BEGIN{printf \"%.0f\", $mt/1024}")
                [ "$first" -eq 1 ] && first=0 || printf " "
                printf "${B}%s${N}:%s%%/${G}%s${N}/%sG" "$idx" "$util" "$mem_gb" "$tot_gb"
            done <<< "$GPU_INFO"
            echo ""
        fi
    fi
fi

echo -e "${D}──────────────────────────────────────────────────────────────────────────────────────${N}"
echo -e "  ${D}./v.sh -w${N}"
echo ""
