#!/bin/bash
# CoEvol-NO on Pipe
# Usage: bash scripts/run_pipe.sh /path/to/data [gpu]

DATA_PATH=${1:?"Usage: bash scripts/run_pipe.sh /path/to/data [gpu]"}
GPU=${2:-0}

python tasks/pde_benchmarks/run.py \
    --config configs/pde/pipe.yaml \
    --data_path "$DATA_PATH" \
    --gpu "$GPU"
