#!/bin/bash
# CoEvol-NO on Car Design (ShapeNetCar)
# Usage: bash scripts/run_car.sh /path/to/data [gpu]

DATA_PATH=${1:?"Usage: bash scripts/run_car.sh /path/to/data [gpu]"}
GPU=${2:-0}

python tasks/car_design/run.py \
    --config configs/car/default.yaml \
    --data_path "$DATA_PATH" \
    --gpu "$GPU"
