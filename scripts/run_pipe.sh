#!/bin/bash
# PDE benchmark: Pipe Flow
# Usage: bash scripts/run_pipe.sh [DATA_PATH] [GPU_ID]

DATA_PATH=${1:-"data/AI4PDE/pipe"}
GPU=${2:-0}

python tasks/pde_benchmarks/run.py \
    --config configs/pde/pipe.yaml \
    --data_path $DATA_PATH \
    --gpu $GPU
