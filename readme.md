# Ozaki Scheme - Triton 原型实现

通过低精度 Tensor Core 模拟高精度矩阵乘法的 Triton 实现。

## 1. 功能概述

Ozaki Scheme 把高精度矩阵分解为多个低精度分片,通过多次低精度 GEMM 在外层高精度累加,达到接近高精度 GEMM 的效果。

**两种典型应用场景**

| 场景 | 输入 dtype | 分片 dtype | 目的 |
|---|---|---|---|
| 低精模拟高精 | FP64 / FP32 | FP16 / INT8 | 用低精 Tensor Core 算力换吞吐 |
| 低精输入高精中间 | FP16 / BF16 | FP32 | 避免直接低精 GEMM 的精度损失 |

**实现特性**

- 全 Triton kernel:融合 split、融合 scale epilogue、跨 slice pair 单 kernel 累加
- 支持任意 broadcastable 前置 batch 维,语义同 `torch.matmul`
- 多精度组合:输入 FP64/FP32/FP16/BF16 × 分片 FP32/FP16/BF16/INT8

---

## 2. 算法

```
C = A @ B

Step 1  Split:    A → [(A_1, sa_1), ..., (A_l, sa_l)]   每个分片 alpha 位有效尾数
                  B → [(B_1, sb_1), ..., (B_l, sb_l)]
Step 2  Compute:  C_ij = tl.dot(A_i, B_j)               低精度 GEMM, 硬件累加
Step 3  Sum:      C = Σ_ij sa_i * sb_j * C_ij           外层高精度累加
```

**关键参数**

| 参数 | 含义 |
|---|---|
| `num_splits` | 分片数量,精度随 `num_splits` 提升,计算量 `O(num_splits^2)` |
| `alpha` | 每个分片的有效尾数位数,由 K 和硬件累加器位宽自动算出 |

`alpha = floor((acc_bits - log2(K)) / 2)`,其中 `acc_bits` 是 `tl.dot` 硬件累加器位数 (FP32→23,INT32→31),且不超过分片 dtype 自身的有效位数。`log2(K)` 项是为单次 `tl.dot` 内部 K 维归约预留的余量。

**精度层次**

精度由三个独立的累加位置共同决定:

1. **`tl.dot` 内部累加** (硬件): Tensor Core 给定,FP16/BF16 → FP32,INT8 → INT32。决定 `alpha` 上界,不可改。
2. **分片间累加** (软件): kernel 内 `out += dot_result * sa * sb`,FP64 输入用 FP64,其余 FP32,原则是取输入精度和分片精度中较高的那个。决定 `num_splits^2` 个 slice pair 求和的精度。
3. **输出 cast**: store 时转回用户的 input_dtype。

FP64 输入要达到 FP64 精度通常需要 `num_splits ≥ 6`,因为单分片只能携带 ~5–8 位有效尾数,需要多片在 FP64 外层累加里叠加。

---

## 3. 代码结构

### 核心模块 `ozaki_triton.py`

| 函数 / Kernel | 功能 |
|---|---|
| `_select_dot_accum_dtype` | 根据分片 dtype 选 `tl.dot` 硬件累加器 (FP32 / INT32) |
| `_select_out_accum_dtype` | 根据输入 dtype 选外层软件累加器 (FP64 / FP32) |
| `compute_split_bits` | 计算每个分片可安全使用的 alpha |
| `split_matrix_kernel` | Triton kernel: 输入矩阵 → `num_splits` 个分片 + scale |
| `split_matrix` | Python 封装,调用 split kernel |
| `matmul_kernel` | Triton kernel: 融合 `num_splits^2` 个 GEMM + scale + 外层累加 |
| `_canonicalize_batch` | 统一 broadcast batch 维并拍平为 3D |
| `ozaki_matmul` | 主入口:batch 处理 → split → matmul → reshape 输出 |

### 设计要点

- **Split kernel 融合**:整行一次性加载到寄存器,`num_splits` 次迭代在 SRAM 内完成,residual 不写回 HBM。
- **跨 slice pair 融合**:K 主循环内维护 `num_splits^2` 个累加器,A/B 读取从 `num_splits^2` 次降至 `2 × num_splits` 次,C 只写一次。
- **Scale 融合**:scale 提到 K 循环外预加载,在 epilogue 应用。
- **多精度统一接口**:所有 dtype 决策通过 `_select_*` 函数集中管理,kernel 内通过 constexpr 静态特化,无运行时分支。
- **Autotune 维度区分**:`IS_INT_SLICE`、`IS_FP64_OUT`、`IS_FP64_RESIDUAL` 进入 autotune key,不同精度组合各自寻优。

### 约束

- `BLOCK_K ≥ K` 且为 2 的幂,且 `BLOCK_K ≤ 8192`(整行装入 SRAM 的限制)。
- A 和 B 的 dtype 必须相同。

---

## 4. 使用示例

```python
import torch
from ozaki_triton import ozaki_matmul

# 场景 A1: FP32 输入, FP16 分片 (经典 Ozaki)
C = ozaki_matmul(A_fp32, B_fp32, num_splits=3, slice_dtype=torch.float16)

# 场景 A2: FP32 输入, INT8 分片 (更高吞吐)
C = ozaki_matmul(A_fp32, B_fp32, num_splits=4, slice_dtype=torch.int8)

# 场景 A3: FP64 输入, INT8 分片
C = ozaki_matmul(A_fp64, B_fp64, num_splits=6, slice_dtype=torch.int8)

# 场景 B: FP16 输入, FP32 分片 (避免直接 FP16 GEMM 损失)
# num_splits=1 即可, 单个 FP32 分片可无损包含 FP16 输入
C = ozaki_matmul(A_fp16, B_fp16, num_splits=1, slice_dtype=torch.float32)
```

---

## 5. 测试

### 环境

- NVIDIA GPU (FP16 Tensor Core,推荐 A100 / H100)
- PyTorch + Triton

```bash
pip install torch triton
```

### 运行

```bash
source run_test.sh
```

输出文件:`test_accuracy.csv`、`test_performance.csv`、`test_summary.txt`、`test_summary.png`

可在 `run_test.sh` 中调整的环境变量:

- `TRITON_PRINT_AUTOTUNING=1`:打印 autotune 选择的最优 config
- `TRITON_DISABLE_CACHE=1`:禁用 autotune 缓存,每次重新调优

### 测试覆盖

**精度测试**:5 种精度组合 (FP32→FP16、FP32→INT8、FP64→FP16、FP64→INT8、FP16→FP32) × 多个 `num_splits`,对比 Naive 低精 GEMM 和 Ozaki 各档。

**性能测试**:5 种精度组合 × 3 种矩阵规模 (512 / 1024 / 2048),指标为执行时间和相对 FP32 GEMM 的 speedup。

### 预期精度参考 (FP32 输入,FP16 分片)

| 方法 | 相对误差 | 最大绝对误差 |
|---|---|---|
| Naive FP16 | ~1e-3 | ~1e-1 |
| Ozaki splits=2 | ~1e-4 | ~1e-2 |
| Ozaki splits=3 | ~1e-5 | ~1e-3 |
| Ozaki splits=4 | ~1e-6 | ~1e-4 |

---

## 6. 优化历史

### 6.1 性能优化

#### 融合 split_matrix 为单 Triton kernel  `9603259`

将原本 Python 循环中每个分片独立的 `amax` / `frexp` / `ldexp` / 除法 / 转 FP16 / 残差更新等步骤融合进单个 kernel。整行 A 一次性加载,`num_splits` 次迭代在 SRAM 内完成,residual 不写回 HBM。用 `tl.exp2(tl.ceil(tl.log2(...)) - ALPHA)` 替代 `frexp + ldexp`,用 `tl.static_range` 编译期展开 split 循环。

收益:分片阶段从 `O(num_splits × 6)` 个 kernel 压缩成 1 个,HBM 流量从多次往返降至 1 读 + 1 写。

#### 融合 scale 进 GEMM epilogue  `5e2e7c3`

将 `scale_a[:, None] * scale_b[None, :]` 融合进 matmul kernel 的写回阶段,用 `tl.atomic_add` 直接累加到 C,省掉临时张量和单独的 add kernel。配套加 `reset_to_zero=['C_ptr']` 防止 autotune benchmark 污染传入的 C。

收益:每次 GEMM 省掉 2 个 elementwise kernel。

#### 跨 slice pair 融合进单 kernel  `1695f5c`

将 `num_splits^2` 次独立 GEMM 调用融合进单 kernel。K 主循环内一次性加载所有分片的 A/B tile,维护一个累加器进行 `num_splits^2` 次累加计算。

- A/B 的 HBM 读取次数:`num_splits^2` → `2 × num_splits` (splits=3 时 9 → 6)
- C 只写一次,省掉 `num_splits^2 - 1` 次中间累加读写
- 移除了原来非融合情况下所需的 `tl.atomic_add` 和 `reset_to_zero`(单 kernel 内不再需要)

Kernel 签名变化:`A_slices` 形状 `(num_splits, M, K)`,`B_slices` 形状 `(num_splits, K, N)`,scales 同步带 split 维,`NUM_SPLITS` 作为 constexpr。scale 提到 K 循环外预加载。

### 6.2 功能扩展

#### 支持 broadcastable batch 维  `533e8f2`

通过 `_canonicalize_batch` 统一 broadcast 逻辑:`torch.broadcast_shapes` 算出统一 batch shape,`expand` 零拷贝扩展,`reshape + contiguous` 拍平为 3D。kernel 加 batch 维 program_id,通过 stride 偏移指针。语义和 `torch.matmul` 一致,纯 2D 输入也兼容。

#### 多精度支持  `e90f7f2`

支持输入 FP64/FP32/FP16/BF16 × 分片 FP32/FP16/BF16/INT8 的任意组合。

- 三个累加位置的 dtype 通过 `_select_dot_accum_dtype` 和 `_select_out_accum_dtype` 集中管理
- `compute_split_bits` 参数化,根据 `dot_accum_dtype` 选 acc_bits
- `matmul_kernel` 新增 `OUT_ACCUM_DTYPE` constexpr,FP64 输入时外层累加器为 FP64,确保跨 slice pair 求和无损
- INT8 路径在 split kernel 里加了 round-half-away-from-zero + clamp;matmul kernel 里 `tl.dot` 显式 `out_dtype=tl.int32`,转 FP32 后乘 scale
- Autotune key 加入 `IS_INT_SLICE`、`IS_FP64_OUT`、`IS_FP64_RESIDUAL`,不同精度组合各自寻优

### 6.3 待办

- [ ] FP64 路径的 tile 范围进一步调优 (FP64 累加器寄存器占用是 FP32 的 2×)

---

## 参考文献

- Ozaki et al., *Error-free transformations of matrix multiplication by using fast routines of matrix multiplication and its applications*, Numerical Algorithms, 2012
- Ootomo et al., *DGEMM on Integer Matrix Multiplication Unit*, IJHPCA, 2024
- Ozaki et al., *Ozaki Scheme II (CRT-based)*, arXiv:2504.08009, 2025
