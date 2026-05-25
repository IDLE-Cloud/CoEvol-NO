#!/bin/bash
# Car design (ShapeNetCar)
# Usage: bash scripts/run_car.sh [DATA_PATH] [GPU_ID]

DATA_PATH=${1:-"data/ShapeNetCar"}
GPU=${2:-0}

python tasks/car_design/run.py \
    --config configs/car/default.yaml \
    --data_path $DATA_PATH \
    --gpu $GPU
