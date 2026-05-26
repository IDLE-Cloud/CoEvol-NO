#!/bin/bash
# CoEvol-NO on Elasticity
# Usage: bash scripts/run_elasticity.sh /path/to/data [gpu]

DATA_PATH=${1:?"Usage: bash scripts/run_elasticity.sh /path/to/data [gpu]"}
GPU=${2:-0}

python tasks/pde_benchmarks/run.py \
    --config configs/pde/elasticity.yaml \
    --data_path "$DATA_PATH" \
    --gpu "$GPU"
