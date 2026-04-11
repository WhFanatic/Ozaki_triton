#!/bin/bash
# 默认不打印 autotune 配置，不禁用缓存
# 需要时取消下方注释
# export TRITON_PRINT_AUTOTUNING=1
# export TRITON_DISABLE_CACHE=1

python test.py
