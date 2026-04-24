export TRITON_PRINT_AUTOTUNING=1
export TRITON_DISABLE_CACHE=1
export CUDA_VISIBLE_DEVICES=5

python tune.py > tune_log.txt
# sed -i '/^Autotuning kernel/d' tune_log.txt