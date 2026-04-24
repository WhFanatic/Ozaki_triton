#!/bin/bash

export TRITON_PRINT_AUTOTUNING=1
export TRITON_DISABLE_CACHE=0
export CUDA_VISIBLE_DEVICES=5

NCU_PATH="/usr/local/cuda-12.1/bin/ncu"
PYTHON_PATH="/home/amax/l50044990/conda_envs/whn/bin/python"
OUTPUT_FILE="report_ozaki"

sudo "$NCU_PATH" \
    --replay-mode kernel \
    --target-processes all \
    --profile-from-start no \
    --set basic \
    -o "$OUTPUT_FILE" \
    "$PYTHON_PATH" perf.py

"$NCU_PATH" -i "$OUTPUT_FILE.ncu-rep" > perf_summary.txt
