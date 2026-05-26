#!/bin/bash
# CoEvol-NO on Navier-Stokes
# Usage: bash scripts/run_ns.sh /path/to/data [gpu]

DATA_PATH=${1:?"Usage: bash scripts/run_ns.sh /path/to/data [gpu]"}
GPU=${2:-0}

python tasks/pde_benchmarks/run.py \
    --config configs/pde/ns.yaml \
    --data_path "$DATA_PATH" \
    --gpu "$GPU"
