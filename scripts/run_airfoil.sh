#!/bin/bash
# Airfoil design (AirfRANS)
# Usage: bash scripts/run_airfoil.sh [DATA_PATH] [GPU_ID]

DATA_PATH=${1:-"data/Dataset"}
GPU=${2:-0}

python tasks/airfoil_design/run.py \
    --config configs/airfoil/full.yaml \
    --data_path $DATA_PATH \
    --gpu $GPU
