#!/bin/bash
# PDE benchmark: Elasticity
# Usage: bash scripts/run_elasticity.sh [DATA_PATH] [GPU_ID]

DATA_PATH=${1:-"data/AI4PDE"}
GPU=${2:-0}

python tasks/pde_benchmarks/run.py \
    --config configs/pde/elasticity.yaml \
    --data_path $DATA_PATH \
    --gpu $GPU
