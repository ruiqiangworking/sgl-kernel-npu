"""
固定用例 dispatch 正确性测试（独立脚本）

使用预设的、确定性的 topk_idx 和 x（每个 rank 的 token 值全等于 rank 编号），
在本地通过纯数学公式推导出各 rank 应收到的 recv_x 期望值（不依赖任何跨 rank 通信），
再与实际 dispatch 计算结果在 CPU numpy 上逐元素对比，验证 dispatch 的正确性。

topk_idx 生成公式（所有 rank 均已知，因此可本地重建任意 src_rank 的 topk_idx）：
    start = (rank * num_tokens + token_idx) % num_experts
    topk  = [start+0, start+1, ..., start+num_topk-1] % num_experts

Usage:
    python test_fixed_dispatch.py \\
        --num-processes 16 \\
        --num-tokens 1024 \\
        --hidden 7168 \\
        --num-topk 8 \\
        --num-experts 256
"""

import argparse
import os

# noinspection PyUnresolvedReferences
import deep_ep
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch_npu
from utils import init_dist


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

def _per_token_cast_back_shmem(
    x_fp8: torch.Tensor,
    x_scales: torch.Tensor,
    local_rank: int,
) -> torch.Tensor:
    """将 FP8 + scales 还原为 bfloat16 并写入 symmetric memory（与主测试文件保持一致）。"""
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    recv_x_bf16 = (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)
    device = torch.device(f"npu:{local_rank}")
    shmem_recv_x = symm_mem.empty(recv_x_bf16.shape, dtype=torch.bfloat16, device=device)
    return shmem_recv_x


# --------------------------------------------------------------------------- #
# Core test
# --------------------------------------------------------------------------- #

def run_fixed_dispatch(
    local_rank: int,
    num_local_ranks: int,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    buffer: deep_ep.Buffer,
    rank: int,
    num_ranks: int,
) -> None:
    """
    每个进程执行一次固定输入的 dispatch 并验证结果。

    Args:
        local_rank:      当前进程在本机上的编号
        num_local_ranks: 本机总进程数
        num_tokens:      每个 rank 的 token 数量
        hidden:          隐藏维度
        num_topk:        每个 token 选择的 expert 数量
        num_experts:     总 expert 数量
        buffer:          deep_ep.Buffer 实例
        rank:            全局 rank 编号
        num_ranks:       全局总 rank 数量
    """
    assert num_experts % num_ranks == 0, "num_experts 必须能被 num_ranks 整除"
    experts_per_rank = num_experts // num_ranks

    if local_rank == 0:
        print(
            f"[fixed] num_tokens={num_tokens}, hidden={hidden}, "
            f"num_topk={num_topk}, num_experts={num_experts}, num_ranks={num_ranks}",
            flush=True,
        )
        print("[fixed] Running fixed dispatch test ...", flush=True)

    # ------------------------------------------------------------------ #
    # 1. 构造固定、确定性的 topk_idx（NPU 张量）
    #    token t 在 rank r 上选择 num_topk 个连续 expert，
    #    起点 = (rank * num_tokens + t) % num_experts，保证无重复且覆盖多个 rank
    # ------------------------------------------------------------------ #
    t_indices = torch.arange(num_tokens, dtype=torch.int64, device="npu")
    start_experts = (rank * num_tokens + t_indices) % num_experts      # (num_tokens,)
    k_offsets = torch.arange(num_topk, dtype=torch.int64, device="npu")  # (num_topk,)
    fixed_topk_idx = (
        start_experts.unsqueeze(1) + k_offsets.unsqueeze(0)
    ) % num_experts                                                      # (num_tokens, num_topk)

    # ------------------------------------------------------------------ #
    # 2. 构造固定的 x：x[t, :] = float(rank)
    #    bfloat16 可精确表示 0–255 的整数，来源 rank 编号 ≤ 255，验证时无精度损失
    # ------------------------------------------------------------------ #
    fixed_x = (
        torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * rank
    )
    fixed_topk_weights = torch.ones(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    )

    # ------------------------------------------------------------------ #
    # 3. 调用 get_dispatch_layout 获取 layout 信息
    # ------------------------------------------------------------------ #
    config = deep_ep.Config(24, 8, 256)
    (
        fixed_num_tokens_per_rank,
        _,
        fixed_num_tokens_per_expert,
        fixed_is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(fixed_topk_idx, num_experts)

    # ------------------------------------------------------------------ #
    # 4. 实际执行 dispatch
    # ------------------------------------------------------------------ #
    (
        actual_recv_x,
        _actual_recv_topk_idx,
        _actual_recv_topk_weights,
        _actual_recv_num_tokens_per_expert_list,
        _handle,
        _event,
    ) = buffer.dispatch(
        x=fixed_x,
        num_tokens_per_rank=fixed_num_tokens_per_rank,
        is_token_in_rank=fixed_is_token_in_rank,
        num_tokens_per_expert=fixed_num_tokens_per_expert,
        config=config,
        topk_idx=fixed_topk_idx,
        topk_weights=fixed_topk_weights,
    )
    if isinstance(actual_recv_x, tuple):
        actual_recv_x = _per_token_cast_back_shmem(*actual_recv_x, local_rank)

    # ------------------------------------------------------------------ #
    # 5. 模拟 dispatch 运行推导 expected recv_x（CPU numpy，节省 NPU 显存）
    #
    #    recv_x 的行顺序：expert 升序（外层）→ src_rank 升序（中层）→ token 升序（内层）
    #
    #    步骤：
    #    (a) 重建整个通讯域所有 rank 的 topk_idx（与构造 fixed_topk_idx 的公式相同）。
    #    (b) 对本 rank 负责的每个 expert，遍历每个 src_rank，找出该 src_rank 中哪些
    #        token 的 topk_idx 包含该 expert（token 升序），得到 token_per_rank。
    #    (c) 因为 fixed_x[t, :] = float(src_rank)，对应块用 src_rank 值填充。
    # ------------------------------------------------------------------ #
    my_expert_start = rank * experts_per_rank
    my_expert_end = (rank + 1) * experts_per_rank

    # (a) 重建所有 rank 的 topk_idx（CPU numpy）
    t_np = np.arange(num_tokens, dtype=np.int64)          # (num_tokens,)
    k_np = np.arange(num_topk, dtype=np.int64)            # (num_topk,)
    all_topk_idx: list[np.ndarray] = []
    for src_rank in range(num_ranks):
        start_np = (src_rank * num_tokens + t_np) % num_experts  # (num_tokens,)
        topk_np = (start_np[:, None] + k_np[None, :]) % num_experts  # (num_tokens, num_topk)
        all_topk_idx.append(topk_np)

    # (b)(c) 按 expert 升序 → src_rank 升序 → token 升序 填充 expected_np
    expected_recv_blocks: list[np.ndarray] = []
    expected_src_info: list[tuple[int, int, int]] = []  # (expert_abs, src_rank, cnt)
    for expert_abs in range(my_expert_start, my_expert_end):
        for src_rank in range(num_ranks):
            topk_np = all_topk_idx[src_rank]               # (num_tokens, num_topk)
            # 找出 topk 中包含 expert_abs 的 token（token 索引已天然升序）
            mask = np.any(topk_np == expert_abs, axis=1)   # (num_tokens,)
            cnt = int(mask.sum())
            expected_src_info.append((expert_abs, src_rank, cnt))
            if cnt > 0:
                # fixed_x[t, :] = float(src_rank)，故整块填充 src_rank 值
                expected_recv_blocks.append(
                    np.full((cnt, hidden), float(src_rank), dtype=np.float32)
                )

    expected_np = (
        np.concatenate(expected_recv_blocks, axis=0)
        if expected_recv_blocks
        else np.zeros((0, hidden), dtype=np.float32)
    )
    expected_total = expected_np.shape[0]

    # ------------------------------------------------------------------ #
    # 6. 将 actual_recv_x 搬到 CPU，转为 float32 numpy，对比 shape
    # ------------------------------------------------------------------ #
    actual_np = actual_recv_x.to(torch.float32).cpu().numpy()
    del actual_recv_x  # 及时释放 NPU symmetric tensor

    assert actual_np.shape == expected_np.shape, (
        f"[fixed] recv_x shape mismatch on rank {rank}:\n"
        f"  Expected shape: {expected_np.shape}\n"
        f"  Actual shape:   {actual_np.shape}\n"
        f"  src_info (src_rank, expected_tokens): {expected_src_info}"
    )

    # ------------------------------------------------------------------ #
    # 7. 在 CPU 上用 numpy.allclose 逐元素对比内容（节省 NPU 显存）
    # ------------------------------------------------------------------ #
    match = np.allclose(actual_np, expected_np, atol=1e-2, rtol=0)
    assert match, (
        f"[fixed] recv_x content mismatch on rank {rank}:\n"
        f"  Max abs diff: {np.abs(actual_np - expected_np).max():.4f}\n"
        f"  actual_recv_x[:5,0]:   {actual_np[:5, 0].tolist()}\n"
        f"  expected_recv_x[:5,0]: {expected_np[:5, 0].tolist()}"
    )

    print(
        f"{rank=}, fixed dispatch test passed | "
        f"total_recv={expected_total} | "
        f"src_info={expected_src_info}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Process entry point
# --------------------------------------------------------------------------- #

def test_loop(
    local_rank: int,
    num_local_ranks: int,
    args: argparse.Namespace,
) -> None:
    rank, num_ranks, _group = init_dist(local_rank, num_local_ranks)

    print(f"[Rank {rank} | Local rank {local_rank}] Initializing buffer...", flush=True)
    buffer = deep_ep.Buffer(
        _group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
    )
    print(f"[Rank {rank}] Buffer created OK.", flush=True)

    run_fixed_dispatch(
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        num_experts=args.num_experts,
        buffer=buffer,
        rank=rank,
        num_ranks=num_ranks,
    )

    dist.barrier()
    dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixed-input dispatch correctness test (standalone)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to spawn (default: 16)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=1024,
        help="Number of tokens per rank (default: 1024)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=7168,
        help="Hidden dimension size (default: 7168)",
    )
    parser.add_argument(
        "--num-topk",
        type=int,
        default=8,
        help="Number of top-k experts per token (default: 8)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Total number of experts (default: 256)",
    )
    args = parser.parse_args()

    torch.multiprocessing.spawn(
        test_loop,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
