#!/bin/bash
# PDE benchmark: Darcy Flow
# Usage: bash scripts/run_darcy.sh [DATA_PATH] [GPU_ID]

DATA_PATH=${1:-"data/AI4PDE"}
GPU=${2:-0}

python tasks/pde_benchmarks/run.py \
    --config configs/pde/darcy.yaml \
    --data_path $DATA_PATH \
    --gpu $GPU
