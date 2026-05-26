#!/bin/bash
# CoEvol-NO on Darcy Flow
# Usage: bash scripts/run_darcy.sh /path/to/data [gpu]

DATA_PATH=${1:?"Usage: bash scripts/run_darcy.sh /path/to/data [gpu]"}
GPU=${2:-0}

python tasks/pde_benchmarks/run.py \
    --config configs/pde/darcy.yaml \
    --data_path "$DATA_PATH" \
    --gpu "$GPU"
