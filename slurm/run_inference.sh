#!/bin/bash
#SBATCH --job-name=reasoning_gt
#SBATCH --partition=base_suma_rtx3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --time=12:00:00
#SBATCH --output=output/reasoning_%j.out
#SBATCH --error=output/reasoning_%j.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "GPU Count: $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"
echo "Start: $(date)"
echo "=========================================="

mkdir -p output

source ar1_venv/bin/activate
source env.sh

cd /home/tropity24/AlpamayoR1_Copy
python -u -m alpamayo_r1.generate_reasoning_gt

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
