#!/bin/bash

export TRITON_PRINT_AUTOTUNING=0
export TRITON_DISABLE_CACHE=0
export CUDA_VISIBLE_DEVICES=5

python test.py
