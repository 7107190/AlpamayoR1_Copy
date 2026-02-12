#!/bin/bash
# GPU monitoring script for SLURM jobs
# Usage: ./monitor_gpu.sh JOBID

JOBID=$1

if [ -z "$JOBID" ]; then
    echo "Usage: $0 JOBID"
    exit 1
fi

# Get node name
NODE=$(squeue -j $JOBID -h -o "%N")

if [ -z "$NODE" ]; then
    echo "Job $JOBID not found or not running"
    exit 1
fi

echo "Job $JOBID is running on node: $NODE"
echo "Connecting to monitor GPU..."
echo "Press Ctrl+C to exit"
echo ""

# Run nvidia-smi in watch mode on the node
srun --jobid=$JOBID --pty bash -c "watch -n 2 nvidia-smi"
