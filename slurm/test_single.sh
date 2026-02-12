#!/bin/bash
#SBATCH --job-name=test_single_clip
#SBATCH --partition=base_suma_rtx3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=output/test_single_%j.out
#SBATCH --error=output/test_single_%j.err

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | GPU: $CUDA_VISIBLE_DEVICES"

source ar1_venv/bin/activate
source env.sh

cd /home/tropity24/AlpamayoR1_Copy
python -u -m alpamayo_r1.test_single_clip
