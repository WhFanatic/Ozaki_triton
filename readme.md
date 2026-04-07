# Ozaki Scheme - Triton 原型实现

## 算法原理

Ozaki Scheme 通过将高精度矩阵分解为多个低精度分片，利用低精度硬件（如 FP16/INT8 Tensor Core）的高吞吐量来模拟高精度矩阵乘法。

```
C = A @ B

     Split           Compute (FP16 GEMM)         Sum (FP32)
A → [A1, A2, A3]  ┐
                   ├→ C_ij = A_i @ B_j^T  →  C = Σ scale_i * scale_j * C_ij
B → [B1, B2, B3]  ┘
```

**三步流程：**

1. **Split**：将每个矩阵按行提取尾数高位，逐层剥离为多个 FP16 分片，每个分片附带行级缩放因子
2. **Compute**：计算所有 `sA × sB` 个 FP16 交叉乘积（利用 Triton kernel）
3. **Sum**：将所有乘积乘以对应缩放因子后累加到 FP32 结果中

## 文件说明

- `ozaki_triton.py` — 核心实现：分片 (split_matrix)、Triton matmul kernel、Ozaki 主流程
- `test.py` — 测试脚本：对比 FP32、Naive FP16、Ozaki Scheme 的精度

## 运行

```bash
# 需要 NVIDIA GPU + PyTorch + Triton
pip install torch triton
python test.py
```

## 预期输出

分片越多精度越高，Ozaki (splits=3~4) 的误差应显著低于 naive FP16：

```
方法                         相对误差       最大绝对误差
Naive FP16                  ~1e-3          ~1e-1
Ozaki (splits=2)            ~1e-4          ~1e-2
Ozaki (splits=3)            ~1e-5          ~1e-3
Ozaki (splits=4)            ~1e-6          ~1e-4
```

## 关键参数

| 参数 | 含义 |
|------|------|
| `num_splits` | 分片数量，↑精度↑ 但计算量为 O(splits²) |
| `alpha` | 每个分片的有效尾数位数，由内积维度 K 自动计算 |

## 局限性

- 这是教学原型，未做性能优化（分片在 CPU 端逐层循环）
- 实际部署需要融合 split/sum kernel、K 维分块以控制显存
- 未实现 INT8 分片路径（工业级实现如 ozIMMU 使用 INT8 Tensor Core）

## 参考文献

- Ozaki et al., *Error-free transformations of matrix multiplication*, Numerical Algorithms, 2012
- Ootomo et al., *DGEMM on Integer Matrix Multiplication Unit*, IJHPCA, 2024
- Ozaki et al., *Ozaki Scheme II* (CRT-based), arXiv:2504.08009, 2025
