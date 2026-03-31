# Shmem 算子工作交接文档

> 本文档面向接手 **DeepEP Shmem 算子**相关工作的开发者，涵盖 C++ 接口、编译流程、测试方法、Shmem 内存管理以及同步机制设计。

---

## 目录

1. [C++ 层与 Shmem 相关的 Buffer 接口](#1-c-层与-shmem-相关的-buffer-接口)
2. [编译 deepep_standalone 算子包](#2-编译-deepep_standalone-算子包)
3. [使用 test_intranode_direct.py 测试 Shmem 接口](#3-使用-test_intranode_directpy-测试-shmem-接口)
4. [Shmem 预分配内存实现与内存排布](#4-shmem-预分配内存实现与内存排布)
5. [ShmemSyncFlag 同步机制与算子时序设计](#5-shmemsyncflag-同步机制与算子时序设计)

---

## 1. C++ 层与 Shmem 相关的 Buffer 接口

### 1.1 核心类：`deep_ep::Buffer`

`Buffer` 类定义于 `deepep_standalone/csrc/deepep/deep_ep.hpp`，是所有 MoE All-to-All 通信的统一入口。通过构造时传入的参数和环境变量 `DEEPEP_SHMEM_ENABLE=1` 来决定走 **Shmem 路径** 还是**传统 CAM/HCCL 路径**。

#### 构造函数

```cpp
Buffer(int64_t rank, int64_t num_ranks,
       int64_t num_nvl_bytes, int64_t num_rdma_bytes,
       bool low_latency_mode,
       std::string moe_all_to_all_group_name,
       std::string shmem_server_ipport = "127.0.0.1:11222",
       int64_t hidden = 0,
       int64_t num_experts_hint = 0,
       int64_t num_topk = 0,
       bool use_quant = false,
       int64_t shmem_st_ratio_x = 8);
```

关键参数说明：

| 参数 | 说明 |
|------|------|
| `rank` / `num_ranks` | 当前 rank 编号 / EP 域总 rank 数 |
| `shmem_server_ipport` | Shmem 服务端 IP:Port，用于 `shmem_init` |
| `hidden` | 模型隐藏维度 H（用于预分配 shmem tensor） |
| `num_experts_hint` | Expert 总数 E（用于预分配 shmem tensor） |
| `num_topk` | TopK 值 |
| `use_quant` | 是否启用 INT8/FP8 量化 |

> 当 `DEEPEP_SHMEM_ENABLE=1` 且 `hidden > 0` 且 `num_experts_hint > 0` 时，构造函数会自动调用 `preallocate_shmem_tensors()` 完成 shmem 内存的预分配。

#### 1.2 公开接口一览

以下是 `Buffer` 类通过 pybind11 导出到 Python 的接口（模块名 `deep_ep_cpp`）：

| 接口 | 功能 | Shmem 路径对应算子 |
|------|------|-------------------|
| `get_dispatch_layout()` | 计算 dispatch 路由表：`num_tokens_per_rank`、`num_tokens_per_expert`、`is_token_in_rank` | `aclnnDispatchLayout` |
| `intranode_dispatch()` | **节点内** dispatch：将 token 按 expert 分发到各 rank | Shmem: `aclnnShmemNotifyDispatch` + `aclnnShmemMoeDispatchNormal` |
| `intranode_combine()` | **节点内** combine：将 expert 计算结果按 token 合并回各 rank | Shmem: `aclnnShmemMoeCombineNormal` |
| `internode_dispatch()` | **节点间** dispatch | `aclnnNotifyDispatchA2` + `aclnnDispatchNormalA2` |
| `internode_combine()` | **节点间** combine | `aclnnMoeDistributeCombineA2` |
| `low_latency_dispatch()` | 低延迟模式 dispatch | — |
| `low_latency_combine()` | 低延迟模式 combine | — |
| `fused_deep_moe()` | 融合 MoE 推理（dispatch + FFN + combine 一体化） | — |

#### 1.3 Shmem 路径 vs CAM 路径的分支逻辑

在 `intranode_dispatch` 和 `intranode_combine` 中，通过 `shmem_enable` 成员变量进行分支：

- **Shmem 路径**（`shmem_enable = true`）：
  - 使用 `shmem_ptr`（共享内存指针）作为 `ext_info` 传递给算子
  - Notify 阶段调用 `aclnnShmemNotifyDispatch`（通过 shmem 写 flag 进行跨卡通知）
  - Dispatch 数据搬运调用 `aclnnShmemMoeDispatchNormal`（通过 shmem 直接 DMA 写远端内存）
  - Combine 调用 `aclnnShmemMoeCombineNormal`（从远端 shmem 读取并加权累加）
  - 所有数据 tensor 均从预分配的 shmem 池中获取，**运行时无额外分配**

- **CAM 路径**（`shmem_enable = false`）：
  - 使用 HCCL 通信域进行 All-to-All 通信
  - Notify 调用 `aclnnNotifyDispatch`
  - Dispatch 调用 `aclnnCamMoeDispatchNormal`
  - Combine 调用 `aclnnCamMoeCombineNormal`
  - 数据 tensor 是普通设备内存，每次运行时按实际大小分配

#### 1.4 预分配的 Shmem Tensor 成员

`Buffer` 类内部维护了以下 Shmem Tensor（构造时一次性分配）：

| 成员变量 | Shape | Dtype | 用途 |
|---------|-------|-------|------|
| `c_shmem_num_tokens_per_expert` | `{E}` | kInt | `get_dispatch_layout` 输出 |
| `c_dispatch_shmem_recv_data` | `{R, E}` | kInt | `ShmemNotifyDispatch` 接收数据 |
| `c_shmem_combine_x` | `{max_recv_tokens, H}` | kBFloat16 | combine 输入 / dispatch 输出共享 |
| `c_shmem_expandx_out` | `{max_recv_tokens, H}` | kChar(quant)/kBF16 | dispatch 输出，与 `c_shmem_combine_x` 共享底层内存 |
| `c_shmem_dynamic_scales_out` | `{max_recv_tokens}` | kFloat | 量化模式下的 per-token scale |

---

## 2. 编译 deepep_standalone 算子包

### 2.1 前提准备

确保安装了以下依赖：

- **Ascend CANN 工具链**（含 `ascend-toolkit`）
- **Shmem 库**（`ascend-shmem`）
- **PyTorch** + **torch_npu**
- Python 3.8+

```bash
# 设置环境变量（可选，如未设置则使用默认路径）
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export SHMEM_HOME_PATH=/usr/local/Ascend/shmem/latest

# Source Ascend 环境
source $(dirname $ASCEND_HOME_PATH)/set_env.sh
```

### 2.2 一键构建

```bash
cd deepep_standalone
./build.sh               # 默认 Release 模式，SOC: Ascend910_9382
./build.sh -d            # Debug 模式
./build.sh Ascend910_93  # 指定 SOC 版本
```

### 2.3 构建流程

`build.sh` 内部包含三个阶段：

1. **`build_deepep_kernels`**：编译 Ascend 自定义算子内核
   - 进入 `csrc/deepep/ops/` 目录
   - 调用 `ops/build.sh` 执行 CMake + 交叉编译
   - 产出 `custom_opp*.run` 安装包
   - 自动安装算子到 `python/deep_ep/deep_ep/vendors/` 目录

2. **`build_deepep_adapter`**：编译 C++ pybind 扩展模块
   - 顶层 CMake 构建 `csrc/deepep/` 下的 `deep_ep_cpp.so` 共享库
   - 输出到 `output/lib/`

3. **`make_deepep_package`**：打包 Python wheel
   - 将编译产物复制到 `python/deep_ep/deep_ep/`
   - 调用 `python setup.py bdist_wheel` 生成 `.whl`
   - 输出到 `output/`

### 2.4 安装

```bash
pip3 install output/deep_ep*.whl
# 或强制重新安装
pip3 install --force-reinstall output/deep_ep*.whl
```

### 2.5 单独编译算子内核

如果只修改了 `csrc/deepep/ops/` 下的算子代码，可以单独编译：

```bash
cd csrc/deepep/ops
chmod +x build.sh
./build.sh
```

编译产物位于 `build_out/` 目录。

### 2.6 清理构建

```bash
rm -rf output/
rm -rf csrc/deepep/ops/build_out/
rm -rf python/deep_ep/build/
rm -rf python/deep_ep/dist/
rm -rf build/
```

---

## 3. 使用 test_intranode_direct.py 测试 Shmem 接口

测试文件位于 `tests/python/deepep/test_intranode_direct.py`，直接调用 `deep_ep_cpp.Buffer` 的 C++ pybind 接口，**跳过 Python 层 `deep_ep.Buffer` 包装**。

### 3.1 测试模式

| 模式 | 说明 |
|------|------|
| `fixed` | 规律输入正确性测试：使用本地模拟结果校验 |
| `random` | 随机输入正确性测试：以 HCCLDispatcher 作为参考实现进行对比 |
| `bench` | 性能对比测试：对比 DeepEP Shmem 与 HCCL 的 dispatch/combine 耗时 |
| `profile` | Kineto Profiling：导出 chrome trace 到指定目录 |

### 3.2 运行命令

```bash
# 随机输入正确性测试（默认 16 进程，shmem 模式）
python test_intranode_direct.py \
    --num-processes 16 \
    --num-tokens 1024 \
    --hidden 7168 \
    --num-topk 8 \
    --num-experts 256 \
    --mode random \
    --shmem 1

# 规律输入正确性测试
python test_intranode_direct.py --mode fixed --shmem 1

# 性能基准测试
python test_intranode_direct.py --mode bench --shmem 1

# Kineto Profiling（输出 chrome trace）
python test_intranode_direct.py --mode profile --trace-dir ./traces --num-profile-tests 30

# 关闭 shmem，使用 CAM/HCCL 路径对比
python test_intranode_direct.py --mode random --shmem 0

# 开启量化
python test_intranode_direct.py --mode random --use-quant
```

### 3.3 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-processes` | 16 | 启动的进程数 |
| `--num-tokens` | 1024 | 每 rank token 数上限（实际随机 [1, N]） |
| `--hidden` | 7168 | 隐藏维度 H |
| `--num-topk` | 8 | TopK expert 数 |
| `--num-experts` | 256 | Expert 总数 |
| `--mode` | random | 测试模式：fixed / random / bench / profile |
| `--shmem` | 1 | 0 关闭 shmem，1 开启 shmem |
| `--use-quant` | False | 开启 INT8 量化 |
| `--trace-dir` | `./traces` | Kineto trace 输出目录 |
| `--num-profile-tests` | 30 | Profiling 迭代次数 |

### 3.4 测试流程（以 random 模式为例）

1. 每个进程创建 `deep_ep_cpp.Buffer`（自动初始化 shmem）
2. 生成随机输入 `x` (bf16)、`topk_idx`、`topk_weights`
3. 调用 `get_dispatch_layout` → 本地模拟校验路由表
4. 调用 `intranode_dispatch` → 与 HCCLDispatcher 输出对比
5. 调用 `intranode_combine` → 与 HCCLDispatcher 输出对比
6. 以上循环 1000 次确保稳定性

### 3.5 校验逻辑

- **`verify_dispatch_layout`**：本地按 expert → rank 映射规则模拟计算 `num_tokens_per_rank`、`num_tokens_per_expert`、`is_token_in_rank`，逐元素与算子输出对比
- **`verify_dispatch_expert_tokens`**：all_reduce 后的全局 expert token 分布，对比 dispatch 接收到的 per-expert token 数
- **`verify_combine_local`**：基于 `combined_x ≈ original_x × sum(valid_weights)` 公式校验
- **`verify_dispatch_with_hccl` / `verify_combine_with_hccl`**：与 HCCL All-to-All 实现的输出做浮点对比

---

## 4. Shmem 预分配内存实现与内存排布

### 4.1 实现概述

Shmem 内存管理位于 `deep_ep.cpp` 的 `Buffer` 构造函数和 `preallocate_shmem_tensors()` 方法中。

**初始化流程**：

1. 构造函数检测 `DEEPEP_SHMEM_ENABLE=1`
2. 调用 `shmem_init`（通过 `shmem.hpp` 中的 `internode::init()`）初始化 shmem 运行时，申请 **4GB** 本地显存池
3. 从池中分配 **100MB** 元数据区（`shmem_calloc`）
4. 调用 `preallocate_shmem_tensors()` 从剩余池空间中分配所有数据 tensor

**核心常量**：

```cpp
constexpr size_t SHMEM_LOCAL_MEM_SIZE = 4UL * 1024 * 1024 * 1024;   // 4 GB 总池
constexpr size_t SHMEM_META_DATA_SIZE = 100UL * 1024 * 1024;        // 100 MB 元数据
```

### 4.2 Shmem 内存排布

整个 4GB Shmem 池的布局如下：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SHMEM_LOCAL_MEM_SIZE (4 GB)                     │
├──────────────────────┬──────────────────────────────────────────────┤
│  SHMEM_META_DATA     │  数据区 (可分配给 Tensor)                     │
│  (100 MB)            │                                              │
│                      │  ┌─ num_tokens_per_expert ── E × 4B         │
│  含 shmem 管理结构    │  ├─ dispatch_recv_data ──── R × E × 4B      │
│  和跨卡元数据         │  ├─ combine_x / expandx ── T × H × 2B      │
│                      │  │  (共享同一块内存)                          │
│                      │  └─ dynamic_scales_out ─── T × 4B (仅量化)   │
├──────────────────────┴──────────────────────────────────────────────┤
│  ← 低地址                                          高地址 →         │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Tensor 分配明细表

以 DeepSeek-V3 典型配置（H=7168, E=256, R=16, BF16 无量化）为例：

| Tensor | Shape | Dtype | 大小公式 | 示例大小 |
|--------|-------|-------|---------|---------|
| 元数据保留区 | — | — | 固定 100MB | 100 MB |
| `c_shmem_num_tokens_per_expert` | {E} | kInt32 | E × 4 | 1 KB |
| `c_dispatch_shmem_recv_data` | {R, E} | kInt32 | R × E × 4 | 16 KB |
| `c_shmem_combine_x` (=`expandx_out`) | {T, H} | kBF16 | T × H × 2 | ~3.79 GB |
| `c_shmem_dynamic_scales_out` | {T} | kFloat | T × 4（仅量化） | 0（BF16 模式） |
| **合计** | — | — | — | **~3.89 GB** |

> 其中 T = max_recv_tokens，按公式 `T = ⌊(4GB - 100MB - fixed) / (H × 2 + quant × 4)⌋` 计算。

### 4.4 max_recv_tokens 计算公式

```
fixed_bytes    = E × 4 + R × E × 4 = E × (1 + R) × 4
avail_bytes    = SHMEM_LOCAL_MEM_SIZE - SHMEM_META_DATA_SIZE - fixed_bytes
per_token_bytes = H × 2 + (use_quant ? 4 : 0)
max_recv_tokens = ⌊ avail_bytes / per_token_bytes ⌋
```

### 4.5 expandx_out 与 combine_x 的内存共享

`expandx_out`（dispatch 写入的扩展 token 数据）和 `combine_x`（combine 读取的输入数据）**共享同一块 shmem 内存**，因为它们的生命周期严格不重叠：

1. **Dispatch 阶段**：远端 rank 通过 shmem DMA 写入 `expandx_out`
2. **Expert 计算阶段**：从 `expandx_out` 读取，计算后写入 expert 输出
3. **Combine 阶段**：expert 输出 copy 到 `combine_x`（即同一块内存），远端 rank 从此读取并加权累加

量化模式下，`expandx_out` 为 kChar(int8) 视图（只使用一半字节），`combine_x` 为 kBF16，二者通过 `from_blob` 共享底层指针。

### 4.6 使用 shmem_calculator.xlsx 计算内存池大小

仓库提供了预生成的 Excel 计算器 `deepep_standalone/tools/shmem_calculator.xlsx`，以及生成脚本 `gen_shmem_calculator.py`。

**使用方法**：

1. 打开 `shmem_calculator.xlsx`
2. 在「计算器」sheet 的黄色单元格中填入模型参数：
   - `hidden_size (H)` — 模型隐藏维度（如 7168）
   - `num_experts (E)` — Expert 总数（如 256）
   - `world_size (R)` — EP 并行 rank 数（如 16）
   - `use_quant` — 0 或 1
3. 绿色单元格自动计算：
   - `max_recv_tokens` — shmem 可容纳的最大接收 token 数
   - 各 tensor 大小明细
   - 总 shmem 用量和利用率
4. 如需反向计算（已知 max_recv_tokens → 需要多大池），切到「反向计算器A」sheet

**重新生成计算器**：

```bash
cd deepep_standalone/tools
pip install openpyxl
python gen_shmem_calculator.py
```

---

## 5. ShmemSyncFlag 同步机制与算子时序设计

### 5.1 ShmemSyncFlag 概述

`ShmemSyncFlag` 是一个通用的 **跨卡 flag 同步原语**，定义于 `csrc/deepep/ops/op_kernel/shmem_sync_flag.h`。它利用 shmem 的跨卡可见性，通过在远端共享内存中写入/轮询 flag 值来实现无 HCCL 介入的同步。

### 5.2 Flag 内存布局

每个 rank 的 shmem 中，从 `gva_gm + 100KB`（`FLAG_AREA_BASE`）开始有一块 flag 区域：

```
gva_gm + FLAG_AREA_BASE (100KB):
┌──────────────────────────────────────────────┐
│ Magic Area (24 KB)                           │
│   48 core × 512B/core                        │
│   slot[coreIdx] 存储单调递增的 magic 计数器    │
├──────────────────────────────────────────────┤
│ Barrier PingPong Buffer 0                    │
│   epWorldSize × 32B                          │
├──────────────────────────────────────────────┤
│ Barrier PingPong Buffer 1                    │
│   epWorldSize × 32B                          │
├──────────────────────────────────────────────┤
│ Flag PingPong Buffer 0                       │
│   flag[srcRank][eventID], 每 slot 32B        │
├──────────────────────────────────────────────┤
│ Flag PingPong Buffer 1                       │
│   flag[srcRank][eventID], 每 slot 32B        │
└──────────────────────────────────────────────┘
```

### 5.3 Magic 机制

- 每个 kernel 启动时调用一次 `IncrementMagic()`，读取并递增本 core 的 magic 计数器
- magic 值单调递增：kernel K 对应 magic=M，kernel K+1 对应 magic=M+1
- **PingPong 选择**：`magic % 2` 决定使用哪组 buffer，天然避免连续 kernel 之间的 flag 值冲突
- **Flag 值编码**：`(magic << 32) | phase`，高 32 位为 magic，低 32 位为 phase
- 由于 magic 单调递增，任何过时 kernel 的 flag 值都不会被当前 kernel 误匹配

### 5.4 核心操作

| 操作 | 功能 |
|------|------|
| `SetFlag(destRank, eventID, phase)` | 向目标 rank 的本地 shmem 写入 flag：先 MTE3 fence（确保数据 DMA 完成），再写 flag |
| `WaitFlag(srcRank, eventID, phase)` | 轮询自身本地 shmem，等待源 rank 写入的 flag 匹配 `(magic, phase)` |
| `SetFlagBatch(start, end, eventID, phase)` | 批量向 [start, end) 范围的 rank 写 flag |
| `WaitFlagBatch(start, end, eventID, phase)` | 批量等待 [start, end) 范围 rank 的 flag |
| `SetAllRankCoreFlag(phase)` | 本 core 向所有 rank 写 flag（eventID=coreIdx） |
| `WaitAllRankAllEvent(phase)` | 等待所有 rank 的所有 event slot（rank 按 core 分片） |
| `BarrierAll()` | 全局 barrier：SyncAll → 写 flag → 等 flag → SyncAll |

### 5.5 同步协议

写端（Rank A, core i）与读端（Rank B, core j）的顺序保证：

```
Rank A:                                 Rank B:
  ①  数据 DMA 写入 B 的 shmem
  ②  MTE3 fence (flush 所有写入)
  ③  SetFlag(dest=B, event=i, phase)
       ↓ 写 flag 到 B 的本地 shmem
                                          ④  WaitFlag(src=A, event=i, phase)
                                               ↓ 轮询本地 shmem
                                          ⑤  读取 A 写入的数据（保证可见）
```

**DAG 保证**：`DataWrite(A→B) → MTE3 fence → SetFlag(A→B) → WaitFlag(B←A) → DataRead(B←A)`，无环路依赖。

### 5.6 Phase 约定

在 Shmem 算子中，phase 值的含义约定如下：

| Phase | 值 | 含义 |
|-------|----|------|
| `PHASE_ENTRY` | 1 | Kernel 已进入，上一个 kernel 已完成，输入 tensor 就绪 |
| `PHASE_DONE` | 2 | 当前 kernel 计算/DMA 全部完成，输出 tensor 已写毕 |

### 5.7 Shmem 算子的同步时序设计

以完整的 **Shmem Intranode MoE** 流程为例，涉及 4 个算子依次执行：

```
时间 → 

Rank 0:  DispatchLayout → ShmemNotifyDispatch → ShmemMoeDispatchNormal → ShmemMoeCombineNormal
Rank 1:  DispatchLayout → ShmemNotifyDispatch → ShmemMoeDispatchNormal → ShmemMoeCombineNormal
...
Rank N:  DispatchLayout → ShmemNotifyDispatch → ShmemMoeDispatchNormal → ShmemMoeCombineNormal
```

各算子间的同步关系如下：

#### (1) DispatchLayout → ShmemNotifyDispatch

- **无跨卡同步需求**，同 rank 内按 stream 顺序执行
- DispatchLayout 输出 `num_tokens_per_expert`、`is_token_in_rank` 等路由表
- NotifyDispatch 读取路由表作为输入

#### (2) ShmemNotifyDispatch — 跨卡元数据通知

```
每个 Rank:
  ① IncrementMagic()
  ② 将本 rank 的 per-expert token 计数写入所有远端 rank 的 shmem
  ③ SetFlag → 通知远端 "我的元数据已就绪"
  ④ WaitFlag ← 等待所有远端 rank 的元数据到达
  ⑤ 汇总各 rank 的 token 分布，计算 total_recv_tokens、put_offset、recv_tokens_per_expert 等
```

此阶段通过 `ShmemSyncFlag` 的 flag 机制保证：在汇总统计前，所有 rank 的元数据写入已全局可见。

#### (3) ShmemMoeDispatchNormal — 跨卡数据搬运

```
每个 Rank, 每个 core:
  ① IncrementMagic()
  ② WaitFlag(PHASE_ENTRY) ← 等待 NotifyDispatch 在所有 rank 上完成
      (使用上一个 kernel 的 magic 值)
  ③ 将本 rank 的 token 数据 DMA 写入目标 rank 的 shmem (expandx_out 区域)
  ④ SetFlag(PHASE_DONE) → 通知目标 rank "数据搬运完成"
```

关键点：
- Dispatch 用 `WaitFlagWithMagic(magic - 1, PHASE_DONE)` 等待 NotifyDispatch 完成
- 数据通过 shmem DMA 直接写入远端 rank 的 `expandx_out` 内存
- 完成后通过 `SetFlag(PHASE_DONE)` 通知远端

#### (4) ShmemMoeCombineNormal — 跨卡结果合并

```
每个 Rank, 每个 core:
  ① IncrementMagic()
  ② WaitFlag(PHASE_ENTRY) ← 等待 Dispatch 在所有 rank 上完成
  ③ 将本 rank 的 expert 计算结果写入远端 rank 的 shmem (combine_x 区域)
  ④ SetFlag(PHASE_DONE) → 通知远端 "combine 数据已就绪"
  ⑤ WaitFlag(PHASE_DONE) ← 等待所有远端 rank 的 combine 数据到达
  ⑥ 从本地 shmem 读取所有 rank 写入的 combine 数据，加权求和得到最终输出
```

#### 完整时序图（2 rank 简化示例）

```
Rank 0                                          Rank 1
──────                                          ──────
DispatchLayout                                  DispatchLayout
    │                                               │
    ▼                                               ▼
NotifyDispatch                                  NotifyDispatch
  ├─ write metadata → Rank1 shmem                ├─ write metadata → Rank0 shmem
  ├─ SetFlag(dest=1, PHASE_DONE) ─────────────► │
  │                           ◄───────────────── ├─ SetFlag(dest=0, PHASE_DONE)
  ├─ WaitFlag(src=1, PHASE_DONE) ◄              ├─ WaitFlag(src=0, PHASE_DONE) ◄
  └─ compute put_offset etc.                     └─ compute put_offset etc.
    │                                               │
    ▼                                               ▼
MoeDispatchNormal                               MoeDispatchNormal
  ├─ WaitFlag(NotifyDispatch DONE)               ├─ WaitFlag(NotifyDispatch DONE)
  ├─ DMA token data → Rank1 expandx_out          ├─ DMA token data → Rank0 expandx_out
  ├─ SetFlag(dest=1, PHASE_DONE) ─────────────► │
  │                           ◄───────────────── ├─ SetFlag(dest=0, PHASE_DONE)
  └─ done                                        └─ done
    │                                               │
    ▼  (Expert FFN compute)                         ▼  (Expert FFN compute)
    │                                               │
    ▼                                               ▼
MoeCombineNormal                                MoeCombineNormal
  ├─ WaitFlag(Dispatch DONE)                     ├─ WaitFlag(Dispatch DONE)
  ├─ DMA result → Rank1 combine_x                ├─ DMA result → Rank0 combine_x
  ├─ SetFlag(dest=1, PHASE_DONE) ─────────────► │
  │                           ◄───────────────── ├─ SetFlag(dest=0, PHASE_DONE)
  ├─ WaitFlag(src=1, PHASE_DONE) ◄              ├─ WaitFlag(src=0, PHASE_DONE) ◄
  └─ weighted sum → output                       └─ weighted sum → output
```

### 5.8 ShmemSyncFlag 的同步粒度选择

通过 `Init()` 的 `slotsPerRank` 参数控制同步粒度：

| 粒度 | slotsPerRank | 使用方式 | 适用场景 |
|------|-------------|---------|---------|
| Per-core | `blockNum` (48) | `SetAllRankCoreFlag()` / `WaitAllRankAllEvent()` | Dispatch —— 每个 core 独立处理一组 rank |
| Per-slot | 自定义 N | `SetFlag(dest, slotIdx)` / `WaitFlag(src, slotIdx)` | 任意自定义分片 |
| Per-rank | 1 | `SetFlagBatch(0, worldSize, 0)` / `WaitFlagBatch(0, worldSize, 0)` | NotifyDispatch —— 每 rank 整体通知 |

---

## 附录

### 关键文件索引

| 文件 | 说明 |
|------|------|
| `deepep_standalone/csrc/deepep/deep_ep.hpp` | Buffer 类声明 |
| `deepep_standalone/csrc/deepep/deep_ep.cpp` | Buffer 类实现（含 shmem 预分配） |
| `deepep_standalone/csrc/deepep/shmem.hpp` | shmem 初始化/分配/释放封装 |
| `deepep_standalone/csrc/deepep/pybind_extension.cpp` | pybind11 绑定定义 |
| `deepep_standalone/csrc/deepep/ops/op_kernel/shmem_sync_flag.h` | ShmemSyncFlag 同步原语 |
| `deepep_standalone/csrc/deepep/ops/op_kernel/shmem_notify_dispatch.h` | NotifyDispatch 算子内核 |
| `deepep_standalone/csrc/deepep/ops/op_kernel/shmem_moe_dispatch_normal.h` | Dispatch 算子内核 |
| `deepep_standalone/csrc/deepep/ops/op_kernel/shmem_moe_combine_normal.h` | Combine 算子内核 |
| `deepep_standalone/build.sh` | 一键构建脚本 |
| `deepep_standalone/tools/gen_shmem_calculator.py` | shmem 计算器生成脚本 |
| `deepep_standalone/tools/shmem_calculator.xlsx` | shmem 池容量计算器 |
| `tests/python/deepep/test_intranode_direct.py` | Shmem 接口正确性 / 性能测试 |
