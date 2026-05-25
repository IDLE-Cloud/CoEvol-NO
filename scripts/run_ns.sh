#!/bin/bash
# PDE benchmark: Navier-Stokes
# Usage: bash scripts/run_ns.sh [DATA_PATH] [GPU_ID]

DATA_PATH=${1:-"data/AI4PDE/NavierStokes_V1e-5_N1200_T20.mat"}
GPU=${2:-0}

python tasks/pde_benchmarks/run.py \
    --config configs/pde/ns.yaml \
    --data_path $DATA_PATH \
    --gpu $GPU
