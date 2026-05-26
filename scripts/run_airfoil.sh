#!/bin/bash
# CoEvol-NO on Airfoil Design (AirfRANS)
# Usage: bash scripts/run_airfoil.sh /path/to/Dataset [gpu]

DATA_PATH=${1:?"Usage: bash scripts/run_airfoil.sh /path/to/Dataset [gpu]"}
GPU=${2:-0}

python tasks/airfoil_design/run.py \
    --config configs/airfoil/full.yaml \
    --data_path "$DATA_PATH" \
    --gpu "$GPU"
