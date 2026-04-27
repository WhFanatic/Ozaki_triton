import random
import numpy as np
import torch
import re, csv

from ozaki_triton import split_matrix, cross_gemm, compute_split_bits


def tune_split(num_shapes=50):
    shapes = [(B, M, K) for B in 2**np.arange(4, 5) \
                        for M in 2**np.arange(7, 12) \
                        for K in 2**np.arange(8, 13) \
              if 4 * 128**2 < B * M * K < 32 * 2048**2]
    if num_shapes < len(shapes):
        shapes = random.sample(shapes, num_shapes)

    for shape in shapes:
        B, M, K = shape
        print('tuning shape', shape)

        a = torch.randn((B, M, K), dtype=torch.float64, device='cuda')
        alpha = compute_split_bits(K, slice_dtype=torch.int8, dot_accum_dtype=torch.float32)
        split_matrix(a, num_splits=2, alpha=alpha, slice_dtype=torch.int8)

    def write_csv(in_file='tune_log.txt', out_file='tune_log.csv'):
        markers = ['tuning shape', 'best config selected:']
        with open(in_file, 'r', encoding='utf-8') as f, open(out_file, 'w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out)
            writer.writerow(['M', 'K', 'BLOCK_M', 'BLOCK_K', 'num_warps', 'num_stages'])
            m, k = 0, 0
            for line in f:
                line = line.strip()
                for i, marker in enumerate(markers):
                    if line.startswith(marker):
                        if i == 0:
                            m, k = map(int, re.findall(r'\d+', line)[1:3])
                        elif i == 1:
                            config = dict(re.findall(r'(\w+):\s*([^,]+)', line.strip(marker)))
                            writer.writerow([m, k, config.get('BLOCK_M'), config.get('BLOCK_K'), config.get('num_warps'), config.get('num_stages')])
                        break

    print(flush=True)
    write_csv()


def tune_gemm(num_shapes=50, num_splits=2):
    shapes = [(B, M, K, N) for B in 2**np.arange(4, 5)
                        for M in 2**np.arange(7, 12)
                        for K in 2**np.arange(8, 13)
                        for N in 2**np.arange(7, 12)
              if 4 * 128**2 < B * M * K < 32 * 2048**2]
    if num_shapes < len(shapes):
        shapes = random.sample(shapes, num_shapes)

    for shape in shapes:
        B, M, K, N = shape
        print('tuning shape', (B, M, K, N))

        slice_dtype = torch.int8
        A_slices = torch.randint(-128, 128, (B, num_splits, M, K), dtype=slice_dtype, device='cuda')
        B_slices = torch.randint(-128, 128, (B, num_splits, K, N), dtype=slice_dtype, device='cuda')
        A_scales = torch.randn((B, num_splits, M), dtype=torch.float32, device='cuda')
        B_scales = torch.randn((B, num_splits, N), dtype=torch.float32, device='cuda')

        cross_gemm(A_slices, B_slices, A_scales, B_scales, num_splits=num_splits, output_dtype=torch.float64)

    def write_csv(in_file='tune_log.txt', out_file='tune_log.csv'):
        markers = ['tuning shape', 'best config selected:']
        with open(in_file, 'r', encoding='utf-8') as f, open(out_file, 'w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out)
            writer.writerow(['M', 'K', 'N', 'BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'num_warps', 'num_stages'])
            m, k, n = 0, 0, 0
            for line in f:
                line = line.strip()
                for i, marker in enumerate(markers):
                    if line.startswith(marker):
                        if i == 0:
                            m, k, n = map(int, re.findall(r'\d+', line)[1:4])
                        elif i == 1:
                            config = dict(re.findall(r'(\w+):\s*([^,]+)', line.strip(marker)))
                            writer.writerow([m, k, n, config.get('BLOCK_M'), config.get('BLOCK_N'), config.get('BLOCK_K'), config.get('num_warps'), config.get('num_stages')])
                        break

    print(flush=True)
    write_csv()


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    tune_split()
    # tune_gemm()
