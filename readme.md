# Ozaki Scheme - Triton 原型实现

## 1. 功能概述

Ozaki Scheme 是一种高精度矩阵乘法的软件模拟方案：通过 FP32 精度输入/输出，使用 FP16 Tensor Core 进行低精度计算，达到接近 FP32 的精度。

**适用场景**：仅有 FP16/INT8 Tensor Core 的硬件上模拟高精度 GEMM。

**本项目实现**：
- FP32 输入/输出，FP16 分片计算
- 支持 broadcastable 前置 batch 维，语义同 `torch.matmul`
- 全 Triton kernel 实现：融合 split、融合 scale、跨分片对累加

---

## 2. 算法描述

```
C = A @ B  (A: FP32[M,K], B: FP32[K,N] → C: FP32[M,N])

步骤 1: Split (FP32 → FP16 分片)
    A → [(sA₁, α₁), (sA₂, α₂), ..., (sAₗ, αₗ)]
    B → [(sB₁, β₁), (sB₂, β₂), ..., (sBₗ, βₗ)]
    每个分片提取高位尾数，行级 scale 保证动态范围

步骤 2: Compute (FP16 GEMM)
    计算所有 l² 个交叉乘积：Cᵢⱼ = sAᵢ @ sBⱼᵀ

步骤 3: Sum (FP32 累加)
    C = Σᵢⱼ αᵢ · βⱼ · Cᵢⱼ
```

**关键参数**：
| 参数 | 含义 |
|------|------|
| `num_splits` | 分片数量，↑精度↑ 但计算量 O(splits²) |
| `alpha` | 每个分片的有效尾数位数，由 K 自动计算 |

---

## 3. 代码设计

### 3.1 核心模块 (`ozaki_triton.py`)

| 函数/Kernel | 功能 |
|------------|------|
| `compute_split_bits(n)` | 计算每个分片可安全使用的尾数位数 α |
| `split_matrix_kernel` | Triton kernel：FP32 行 → num_splits 个 FP16 分片 + scale |
| `split_matrix()` | Python 封装：调用 kernel，返回 slices/scales 张量 |
| `matmul_kernel` | Triton kernel：融合 l² 个 GEMM + scale 应用 + FP32 累加 |
| `ozaki_matmul()` | 主入口：处理 batch 维 → split → GEMM → 返回结果 |

### 3.2 设计要点

- **Split kernel 融合**：整行一次性加载，`num_splits` 次迭代在 SRAM 内完成，residual 不写回 HBM
- **跨分片对融合**：K 循环内维护 `num_splits²` 个累加器，A/B 读取从 l² 降至 2l 次
- **Scale 融合**：在 GEMM epilogue 直接应用 row/col scale，免 elementwise kernel
- **Batch 支持**：通过 `_canonicalize_batch` 统一 broadcast 逻辑，拍平为 3D 后处理

### 3.3 约束条件

- `BLOCK_K >= K` 且为 2 的幂（整行装入 SRAM）
- `BLOCK_K <= 8192`（SRAM 容量限制）

---

## 4. 测试与依赖

### 4.1 环境要求

- NVIDIA GPU（支持 FP16 Tensor Core）
- PyTorch + Triton

```bash
pip install torch triton
```

### 4.2 测试方法

```bash
source run_test.sh
```

**环境变量**（可在 run_test.sh 中修改）：
- `TRITON_PRINT_AUTOTUNING=1`：打印 autotune 选择的 kernel 配置
- `TRITON_DISABLE_CACHE=1`：禁用 autotune 缓存，每次重新调优

**测试内容**：
- 精度测试：对比 FP32、Naive FP16、Ozaki (splits=2/3/4) 的相对误差和最大绝对误差
- 性能测试：各方法的执行时间和相对 FP32 的 speedup

### 4.3 预期精度

| 方法 | 相对误差 | 最大绝对误差 |
|------|----------|--------------|
| Naive FP16 | ~1e-3 | ~1e-1 |
| Ozaki (splits=2) | ~1e-4 | ~1e-2 |
| Ozaki (splits=3) | ~1e-5 | ~1e-3 |
| Ozaki (splits=4) | ~1e-6 | ~1e-4 |

---

## 5. 附录

### 5.1 已完成优化

#### 1. 融合 split_matrix 为单 Triton kernel（commit `9603259`）

**实现内容：**
- 将原来 Python 循环中每个分片独立的 `amax`、`frexp`、`ldexp`、除法、转 FP16、残差更新等步骤融合进单个 Triton kernel
- 整行 A 一次性加载到寄存器/SRAM，`num_splits` 次迭代在 SRAM 内完成，residual 始终不写回 HBM
- 使用 `tl.static_range` 实现编译期展开，避免运行时循环开销
- 用 `tl.exp2(tl.ceil(tl.log2(row_max)) - ALPHA)` 替代 `frexp` + `ldexp`

**关键约束：**
- `BLOCK_K >= K` 且为 2 的幂，整行必须能装入 SRAM
- `BLOCK_K <= 8192`，超出则无法缓存

**预期收益：** 分片阶段从 `O(num_splits × 6)` 个 kernel 压缩成 1 个，HBM 流量从多次往返降至 1 读 + 1 写

---

#### 2. 融合 scale 进 GEMM epilogue（commit `5e2e7c3`）

**实现内容：**
- 将 `scale_a[:, None] * scale_b[None, :]` 融合进 matmul kernel 的写回阶段
- 使用 `tl.atomic_add` 直接累加到输出 C，省掉临时张量分配和单独的 add kernel
- 添加 `reset_to_zero=['C_ptr']` 防止 autotune 污染传入的 C

**关键改动：**
- `matmul_kernel` 接收 `scale_a_ptr` 和 `scale_b_ptr`
- 在 accumulator 写回前应用 row/col scaling

**预期收益：** 每次 GEMM 省掉 2 个 elementwise kernel

---

#### 3. 跨分片对融合进单 kernel（commit `1695f5c`）

**实现内容：**
- 将 `sA × sB` 次独立 GEMM 调用融合进单个 kernel
- 在 K 主循环内加载所有分片的 A/B tile，维护 `num_splits²` 个交叉乘积累加器
- A/B 的 HBM 读取次数从 `num_splits²` 降至 `2 × num_splits`
- C 只写一次，省掉中间累加读写

**Kernel 签名变化：**
- `A_slices_ptr`: `(num_splits, M, K)` FP16
- `B_slices_ptr`: `(num_splits, K, N)` FP16
- `A_scales_ptr`: `(num_splits, M)` FP32
- `B_scales_ptr`: `(num_splits, N)` FP32
- `NUM_SPLITS`: constexpr 参数

**实现细节：**
- 预先加载所有分片的 scale 系数到寄存器
- 使用 `tl.static_range` 三重循环：K 循环内嵌套 `i × j` 交叉乘积
- `out += tl.dot(a, b).to(tl.float32) * sa[:, None] * sb[None, :]`

**预期收益：**
- A/B 读取从 9 次降至 6 次（splits=3）
- C 只写一次，省掉 8 次中间累加

---

#### 4. 批处理输入支持（commit `533e8f2`）

**实现内容：**
- 通过 `_canonicalize_batch` 统一 broadcast 逻辑，拍平为 3D 后处理
- 支持 broadcastable 的前置 batch 维，语义同 `torch.matmul`

### 5.2 后续待办

- [ ] **多精度支持**：允许用户自由选择输入精度（FP32/FP64/INT8）和分片计算精度（FP16/BF16/INT8）
  - **注意**：中间分片计算时累加器精度是易错点，需根据输入/分片精度谨慎选择（如 FP16 分片用 FP32 累加，INT8 分片用 INT32 累加）
- [ ] **BF16 变体**：指数范围更大，避免 FP16 clamp 问题
- [ ] **INT8 变体**：配合 INT32 累加器，Tensor Core 吞吐约 2×
- [ ] **双 buffer 预取**：隐藏 HBM 加载延迟

---

## 参考文献

- Ozaki et al., *Error-free transformations of matrix multiplication*, Numerical Algorithms, 2012
- Ootomo et al., *DGEMM on Integer Matrix Multiplication Unit*, IJHPCA, 2024
- Ozaki et al., *Ozaki Scheme II* (CRT-based), arXiv:2504.08009, 2025
