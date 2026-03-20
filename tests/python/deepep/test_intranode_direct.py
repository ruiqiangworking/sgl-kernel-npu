"""
直接调用 deep_ep_cpp.Buffer 的 intranode dispatch/combine 正确性与性能测试。

跳过 Python 层 deep_ep.Buffer 包装，直接使用 C++ pybind 导出的接口完成：
  - get_dispatch_layout
  - intranode_dispatch
  - intranode_combine

测试策略：
  1. 规律输入（fixed）：使用本地模拟运算结果校验。
  2. 随机输入（random）：使用 HCCLDispatcher 作为参考实现进行校验。
  3. 性能对比：使用 utils.bench 对比 HCCLDispatcher 与 deep_ep_cpp.Buffer 的
     dispatch 和 combine 耗时，并打印汇总表格。

Usage:
    python test_intranode_direct.py \\
        --num-processes 16 \\
        --num-tokens 1024 \\
        --hidden 7168 \\
        --num-topk 8 \\
        --num-experts 256
"""

import argparse
import os
import random
import time
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import deep_ep
import deep_ep_cpp

from hccl_dispatcher import HCCLDispatcher
from utils import bench, init_dist, inplace_unique, per_token_cast_back


# =========================================================================== #
#                           参数 / 配置 辅助
# =========================================================================== #

def print_test_config(
    rank: int,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
) -> None:
    """在 rank 0 上打印当前测试配置。"""
    if rank != 0:
        return
    print(
        f"\n{'=' * 70}\n"
        f"[Test Config]\n"
        f"  num_tokens   = {num_tokens}\n"
        f"  hidden       = {hidden}\n"
        f"  num_topk     = {num_topk}\n"
        f"  num_experts  = {num_experts}\n"
        f"  num_ranks    = {num_ranks}\n"
        f"{'=' * 70}",
        flush=True,
    )


def make_config() -> deep_ep_cpp.Config:
    """创建默认的 dispatch/combine 性能调优 Config。"""
    return deep_ep_cpp.Config(
        num_sms=24,
        num_max_nvl_chunked_send_tokens=8,
        num_max_nvl_chunked_recv_tokens=256,
    )


# =========================================================================== #
#                         输入数据构造
# =========================================================================== #

def build_fixed_inputs(
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构造规律性（确定性）输入。

    - topk_idx: token t 选择 num_topk 个连续 expert，起点为
      (rank * num_tokens + t) % num_experts。
    - x: 每行全为 float(rank)。
    - topk_weights: 全 1。

    Returns:
        (x, topk_idx, topk_weights)
    """
    t_indices = torch.arange(num_tokens, dtype=torch.int64, device="npu")
    start = (rank * num_tokens + t_indices) % num_experts
    k_offsets = torch.arange(num_topk, dtype=torch.int64, device="npu")
    topk_idx = (start.unsqueeze(1) + k_offsets.unsqueeze(0)) % num_experts

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * rank
    topk_weights = torch.ones(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    )
    return x, topk_idx, topk_weights


def build_random_inputs(
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构造随机输入。

    - topk_idx: 随机选取 num_topk 个 expert。
    - x: 标准正态 bf16。
    - topk_weights: 标准正态 float32。

    Returns:
        (x, topk_idx, topk_weights)
    """
    scores = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="npu"
    ).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    )
    return x, topk_idx, topk_weights


# =========================================================================== #
#                     deep_ep_cpp.Buffer 调用封装
# =========================================================================== #

def call_get_dispatch_layout(
    buffer: deep_ep_cpp.Buffer,
    topk_idx: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    调用 deep_ep_cpp.Buffer.get_dispatch_layout 并返回三个核心张量。

    Returns:
        (num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank)
    """
    previous_event: Optional[deep_ep_cpp.EventHandle] = None
    (
        num_tokens_per_rank,
        _num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _event,
    ) = buffer.get_dispatch_layout(
        topk_idx,                      # topk_idx
        num_experts,                   # num_experts
        previous_event,                # previous_event
        False,                         # async
        False,                         # allocate_on_comm_stream
    )
    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


def call_intranode_dispatch(
    buffer: deep_ep_cpp.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens_per_rank: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    config: deep_ep_cpp.Config,
    use_quant: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], list, tuple]:
    """
    调用 deep_ep_cpp.Buffer.intranode_dispatch。

    Returns:
        (recv_x, recv_x_scales, num_recv_tokens_per_expert_list, handle)
        其中 handle 是 combine 所需的元组。
        当 use_quant=True 时 recv_x 为 int8, recv_x_scales 为 float32 scales。
    """
    previous_event: Optional[deep_ep_cpp.EventHandle] = None
    (
        recv_x,                         # expandx_out
        recv_x_scales,                  # dynamic_scales_out
        recv_topk_idx,                  # recv_topk_idx (optional)
        recv_topk_weights,              # recv_topk_weights (optional)
        num_recv_tokens_per_expert_list,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        src_idx,                        # expand_idx_out
        send_head,                      # recv_count_
        put_offset,                     # put_offset_
        _event,
    ) = buffer.intranode_dispatch(
        x,                             # x
        None,                          # x_scales
        topk_idx,                      # topk_idx
        topk_weights,                  # topk_weights
        num_tokens_per_rank,           # num_tokens_per_rank
        is_token_in_rank,              # is_token_in_rank
        num_tokens_per_expert,         # num_tokens_per_expert
        0,                             # cached_num_recv_tokens
        None,                          # cached_rank_prefix_matrix
        None,                          # cached_channel_prefix_matrix
        None,                          # dispatch_wait_recv_cost_stats
        1,                             # expert_alignment
        0,                             # num_worst_tokens
        config,                        # config
        previous_event,                # previous_event
        False,                         # async
        False,                         # allocate_on_comm_stream
        use_quant,                     # use_quant
    )
    # 构造 combine 所需的 handle（与 Python Buffer.dispatch 一致）
    handle = (
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        src_idx,
        is_token_in_rank,
        send_head,
        topk_idx,
        topk_weights,
        put_offset,
    )
    return recv_x, recv_x_scales, num_recv_tokens_per_expert_list, handle


def call_intranode_combine(
    buffer: deep_ep_cpp.Buffer,
    recv_x: torch.Tensor,
    handle: tuple,
) -> torch.Tensor:
    """
    调用 deep_ep_cpp.Buffer.intranode_combine。

    Returns:
        combined_x: 合并后的张量。
    """
    (
        _rank_prefix_matrix,
        _channel_prefix_matrix,
        _recv_channel_prefix_matrix,
        src_idx,
        _is_token_in_rank,
        send_head,
        topk_idx,
        topk_weights,
        put_offset,
    ) = handle
    combined_x, _recv_topk_weights, _event = buffer.intranode_combine(
        recv_x,                        # x
        topk_idx,                      # topk_idx
        topk_weights,                  # topk_weights
        src_idx,                       # src_idx
        send_head,                     # send_head
        put_offset,                    # put_offset
        None,                          # combine_send_cost_stats
    )
    return combined_x


# =========================================================================== #
#                     dispatch layout 本地模拟与校验
# =========================================================================== #

def verify_dispatch_layout(
    topk_idx: torch.Tensor,
    num_experts: int,
    num_ranks: int,
    rank: int,
    group: dist.ProcessGroup,
    actual_num_tokens_per_rank: torch.Tensor,
    actual_num_tokens_per_expert: torch.Tensor,
    actual_is_token_in_rank: torch.Tensor,
) -> torch.Tensor:
    """
    本地计算预期 dispatch layout 并与实际输出逐元素对比。

    Returns:
        gbl_num_tokens_per_expert: all_reduce 后的全局 num_tokens_per_expert。
    """
    num_tokens = topk_idx.shape[0]
    experts_per_rank = num_experts // num_ranks
    device = topk_idx.device

    # 计算 rank_idx
    rank_idx = topk_idx // experts_per_rank
    rank_idx = rank_idx.clone()
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # 预期 num_tokens_per_expert
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device=device)
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # 预期 num_tokens_per_rank / is_token_in_rank
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device=device)
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device=device
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count].clone())[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(
            count, dtype=torch.long, device=device
        )
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = (token_idx_in_rank >= 0).to(torch.int)

    assert torch.allclose(actual_num_tokens_per_rank, num_tokens_per_rank), (
        f"num_tokens_per_rank mismatch on rank {rank}"
    )
    assert torch.allclose(actual_num_tokens_per_expert, num_tokens_per_expert), (
        f"num_tokens_per_expert mismatch on rank {rank}"
    )
    assert torch.allclose(actual_is_token_in_rank, is_token_in_rank), (
        f"is_token_in_rank mismatch on rank {rank}"
    )

    print(f"  rank {rank}: dispatch_layout PASSED", flush=True)
    return gbl_num_tokens_per_expert


# =========================================================================== #
#                  dispatch 输出校验（expert token 数量）
# =========================================================================== #

def verify_dispatch_expert_tokens(
    gbl_num_tokens_per_expert: torch.Tensor,
    recv_num_tokens_per_expert_list: list,
    rank: int,
    num_ranks: int,
) -> None:
    """校验 dispatch 返回的 num_recv_tokens_per_expert_list。"""
    expert_token_nums_type = int(os.getenv("MOE_EXPERT_TOKEN_NUMS_TYPE", 1))
    local_expert_token = gbl_num_tokens_per_expert.view(num_ranks, -1)[rank]
    if expert_token_nums_type == 0:
        expected = local_expert_token.cumsum(dim=0).tolist()
    else:
        expected = local_expert_token.tolist()

    assert expected == recv_num_tokens_per_expert_list, (
        f"num_recv_tokens_per_expert_list mismatch on rank {rank}:\n"
        f"  Expected: {expected}\n"
        f"  Actual:   {recv_num_tokens_per_expert_list}"
    )
    print(f"  rank {rank}: dispatch expert tokens PASSED", flush=True)


# =========================================================================== #
#                  combine 输出校验（本地模拟）
# =========================================================================== #

def verify_combine_local(
    combined_x: torch.Tensor,
    original_x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
    rank: int,
    threshold: float = 5e-5,
) -> None:
    """
    使用本地公式校验 combine 结果。

    预期：combined_x ≈ original_x * sum(weights_where_topk_valid)
    """
    weight_sum = topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1)
    expected_x = (original_x.float() * weight_sum)

    actual_np = combined_x.float().cpu().numpy()
    expected_np = expected_x.cpu().numpy()
    passed = np.allclose(actual_np, expected_np, atol=threshold, rtol=threshold)
    assert passed, (
        f"combine output mismatch on rank {rank}: "
        f"max_abs_diff={np.max(np.abs(actual_np - expected_np)):.6e} > threshold={threshold:.6e}"
    )
    print(f"  rank {rank}: combine (local verify) PASSED", flush=True)


# =========================================================================== #
#                  HCCLDispatcher 参考实现校验
# =========================================================================== #

def verify_dispatch_with_hccl(
    recv_x: torch.Tensor,
    hccl_dispatch_out: torch.Tensor,
    rank: int,
    threshold: float = 5e-5,
) -> None:
    """对比 deep_ep_cpp dispatch 输出与 HCCLDispatcher dispatch 输出。"""
    actual_np = recv_x.float().cpu().numpy()
    expected_np = hccl_dispatch_out.float().cpu().numpy()
    actual_np = actual_np[tuple(slice(0, s) for s in expected_np.shape)]
    passed = np.allclose(actual_np, expected_np, atol=threshold, rtol=threshold)
    assert passed, (
        f"dispatch vs HCCLDispatcher mismatch on rank {rank}: "
        f"max_abs_diff={np.max(np.abs(actual_np - expected_np)):.6e}"
    )
    print(f"  rank {rank}: dispatch (HCCL verify) PASSED", flush=True)


def verify_combine_with_hccl(
    combined_x: torch.Tensor,
    hccl_combine_out: torch.Tensor,
    rank: int,
    threshold: float = 5e-5,
) -> None:
    """对比 deep_ep_cpp combine 输出与 HCCLDispatcher combine 输出。"""
    actual_np = combined_x.float().cpu().numpy()
    expected_np = hccl_combine_out.float().cpu().numpy()
    passed = np.allclose(actual_np, expected_np, atol=threshold, rtol=threshold)
    assert passed, (
        f"combine vs HCCLDispatcher mismatch on rank {rank}: "
        f"max_abs_diff={np.max(np.abs(actual_np - expected_np)):.6e}"
    )
    print(f"  rank {rank}: combine (HCCL verify) PASSED", flush=True)


# =========================================================================== #
#                  正确性测试：规律输入
# =========================================================================== #

def test_fixed_correctness(
    buffer: deep_ep_cpp.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    group: dist.ProcessGroup,
    use_quant: bool = False,
) -> None:
    """规律输入正确性测试：使用本地模拟校验。"""
    quant_tag = " (quant)" if use_quant else ""
    if rank == 0:
        print(f"\n--- [Fixed Input{quant_tag}] Correctness Test ---", flush=True)

    x, topk_idx, topk_weights = build_fixed_inputs(
        num_tokens, hidden, num_topk, num_experts, rank
    )
    config = make_config()

    # 1) get_dispatch_layout
    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = (
        call_get_dispatch_layout(buffer, topk_idx, num_experts)
    )
    gbl_num_tokens_per_expert = verify_dispatch_layout(
        topk_idx, num_experts, num_ranks, rank, group,
        num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank,
    )

    # 2) intranode_dispatch
    recv_x, recv_x_scales, recv_expert_list, handle = call_intranode_dispatch(
        buffer, x, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, config,
        use_quant=use_quant,
    )
    verify_dispatch_expert_tokens(
        gbl_num_tokens_per_expert, recv_expert_list, rank, num_ranks
    )

    # 开启量化时对 dispatch 输出进行反量化
    if use_quant:
        recv_x = per_token_cast_back(recv_x, recv_x_scales)

    # 3) intranode_combine
    combined_x = call_intranode_combine(buffer, recv_x, handle)
    threshold = 5e-2 if use_quant else 5e-5
    verify_combine_local(combined_x, x, topk_weights, topk_idx, rank, threshold=threshold)

    dist.barrier()
    if rank == 0:
        print(f"--- [Fixed Input{quant_tag}] All ranks PASSED ---\n", flush=True)


# =========================================================================== #
#                  正确性测试：随机输入
# =========================================================================== #

def test_random_correctness(
    buffer: deep_ep_cpp.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    group: dist.ProcessGroup,
    use_quant: bool = False,
) -> None:
    """随机输入正确性测试：使用 HCCLDispatcher 作为参考实现校验。"""
    quant_tag = " (quant)" if use_quant else ""
    if rank == 0:
        print(f"\n--- [Random Input{quant_tag}] Correctness Test ---", flush=True)

    x, topk_idx, topk_weights = build_random_inputs(
        num_tokens, hidden, num_topk, num_experts
    )
    config = make_config()
    experts_per_rank = num_experts // num_ranks

    # ---- deep_ep_cpp 路径 ----
    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = (
        call_get_dispatch_layout(buffer, topk_idx, num_experts)
    )

    # 同样校验 layout（使用本地模拟）
    gbl_num_tokens_per_expert = verify_dispatch_layout(
        topk_idx, num_experts, num_ranks, rank, group,
        num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank,
    )

    recv_x, recv_x_scales, recv_expert_list, handle = call_intranode_dispatch(
        buffer, x, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, config,
        use_quant=use_quant,
    )
    verify_dispatch_expert_tokens(
        gbl_num_tokens_per_expert, recv_expert_list, rank, num_ranks
    )

    # 开启量化时对 dispatch 输出进行反量化
    if use_quant:
        recv_x = per_token_cast_back(recv_x, recv_x_scales)

    combined_x = call_intranode_combine(buffer, recv_x, handle)

    # ---- HCCLDispatcher 参考路径 ----
    hccl_dispatcher = HCCLDispatcher(group, num_experts, experts_per_rank)
    hccl_dispatch_out = hccl_dispatcher.dispatch(x, topk_idx, topk_weights)
    hccl_combine_out = hccl_dispatcher.combine(hccl_dispatch_out)

    # ---- 对比（量化会引入精度损失，放宽阈值） ----
    threshold = 5e-2 if use_quant else 5e-5
    verify_dispatch_with_hccl(recv_x, hccl_dispatch_out, rank, threshold=threshold)
    verify_combine_with_hccl(combined_x, hccl_combine_out, rank, threshold=threshold)

    dist.barrier()
    if rank == 0:
        print(f"--- [Random Input{quant_tag}] All ranks PASSED ---\n", flush=True)


# =========================================================================== #
#                           性能测试
# =========================================================================== #

def _run_deepep_dispatch(
    buffer: deep_ep_cpp.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens_per_rank: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    config: deep_ep_cpp.Config,
    use_quant: bool = False,
) -> None:
    """deep_ep_cpp dispatch 单次执行（用于 bench）。"""
    call_intranode_dispatch(
        buffer, x, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, config,
        use_quant=use_quant,
    )


def _run_deepep_combine(
    buffer: deep_ep_cpp.Buffer,
    recv_x: torch.Tensor,
    handle: tuple,
) -> None:
    """deep_ep_cpp combine 单次执行（用于 bench）。"""
    call_intranode_combine(buffer, recv_x, handle)


def _run_hccl_dispatch(
    dispatcher: HCCLDispatcher,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    """HCCLDispatcher dispatch 单次执行（用于 bench）。"""
    dispatcher.dispatch(x, topk_idx, topk_weights)


def _run_hccl_combine(
    dispatcher: HCCLDispatcher,
    dispatch_out: torch.Tensor,
) -> None:
    """HCCLDispatcher combine 单次执行（用于 bench）。"""
    dispatcher.combine(dispatch_out)


def bench_performance(
    buffer: deep_ep_cpp.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    group: dist.ProcessGroup,
    num_warmups: int = 10,
    num_tests: int = 100,
    use_quant: bool = False,
) -> None:
    """
    对比 deep_ep_cpp.Buffer 与 HCCLDispatcher 的 dispatch/combine 性能。

    暖机 num_warmups 次，测试 num_tests 次。
    在 rank 0 打印汇总表格。
    """
    quant_tag = " (quant)" if use_quant else ""
    if rank == 0:
        print(f"\n--- Performance Benchmark{quant_tag} ---", flush=True)

    x, topk_idx, topk_weights = build_random_inputs(
        num_tokens, hidden, num_topk, num_experts
    )
    config = make_config()
    experts_per_rank = num_experts // num_ranks

    # ---- 预先执行一次获取 handle / dispatch_out ----
    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = (
        call_get_dispatch_layout(buffer, topk_idx, num_experts)
    )
    recv_x, recv_x_scales, _, handle = call_intranode_dispatch(
        buffer, x, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, config,
        use_quant=use_quant,
    )

    # 开启量化时对 dispatch 输出进行反量化
    if use_quant:
        recv_x = per_token_cast_back(recv_x, recv_x_scales)

    hccl_dispatcher = HCCLDispatcher(group, num_experts, experts_per_rank)
    hccl_dispatch_out = hccl_dispatcher.dispatch(x, topk_idx, topk_weights)

    # ---- bench: deep_ep_cpp dispatch ----
    deepep_dispatch_avg, deepep_dispatch_min, deepep_dispatch_max = bench(
        partial(
            _run_deepep_dispatch,
            buffer, x, topk_idx, topk_weights,
            num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, config,
            use_quant,
        ),
        num_warmups=num_warmups,
        num_tests=num_tests,
    )

    # ---- bench: deep_ep_cpp combine ----
    deepep_combine_avg, deepep_combine_min, deepep_combine_max = bench(
        partial(_run_deepep_combine, buffer, recv_x, handle),
        num_warmups=num_warmups,
        num_tests=num_tests,
    )

    # ---- bench: HCCL dispatch ----
    hccl_dispatch_avg, hccl_dispatch_min, hccl_dispatch_max = bench(
        partial(_run_hccl_dispatch, hccl_dispatcher, x, topk_idx, topk_weights),
        num_warmups=num_warmups,
        num_tests=num_tests,
    )

    # ---- bench: HCCL combine ----
    hccl_combine_avg, hccl_combine_min, hccl_combine_max = bench(
        partial(_run_hccl_combine, hccl_dispatcher, hccl_dispatch_out),
        num_warmups=num_warmups,
        num_tests=num_tests,
    )

    # ---- 汇总并在 rank 0 打印表格 ----
    _print_perf_table(
        rank=rank,
        num_tokens=num_tokens,
        hidden=hidden,
        num_topk=num_topk,
        num_experts=num_experts,
        num_ranks=num_ranks,
        deepep_dispatch_avg=deepep_dispatch_avg,
        deepep_combine_avg=deepep_combine_avg,
        hccl_dispatch_avg=hccl_dispatch_avg,
        hccl_combine_avg=hccl_combine_avg,
    )


def _print_perf_table(
    rank: int,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    deepep_dispatch_avg: float,
    deepep_combine_avg: float,
    hccl_dispatch_avg: float,
    hccl_combine_avg: float,
) -> None:
    """在 rank 0 上输出格式化性能表格。"""
    if rank != 0:
        return

    def _speedup(baseline: float, optimized: float) -> str:
        if optimized <= 0:
            return "N/A"
        ratio = baseline / optimized
        return f"{ratio:.2f}x"

    # 转换为毫秒
    dep_d = deepep_dispatch_avg * 1e3
    dep_c = deepep_combine_avg * 1e3
    hccl_d = hccl_dispatch_avg * 1e3
    hccl_c = hccl_combine_avg * 1e3

    dispatch_speedup = _speedup(hccl_d, dep_d)
    combine_speedup = _speedup(hccl_c, dep_c)

    header = (
        f"\n{'=' * 80}\n"
        f"  Performance Benchmark Results\n"
        f"{'=' * 80}\n"
        f"  Parameters:\n"
        f"    num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, "
        f"num_experts={num_experts}, num_ranks={num_ranks}\n"
        f"    warmup=10, test_iters=100\n"
        f"{'=' * 80}"
    )

    row_fmt = "  {:<20s} {:>16s} {:>16s} {:>16s}"
    sep = "  " + "-" * 68

    table = "\n".join([
        header,
        row_fmt.format("Stage", "DeepEP (ms)", "HCCL (ms)", "Speedup"),
        sep,
        row_fmt.format("Dispatch", f"{dep_d:.4f}", f"{hccl_d:.4f}", dispatch_speedup),
        row_fmt.format("Combine",  f"{dep_c:.4f}", f"{hccl_c:.4f}", combine_speedup),
        sep,
        row_fmt.format(
            "Total",
            f"{dep_d + dep_c:.4f}",
            f"{hccl_d + hccl_c:.4f}",
            _speedup(hccl_d + hccl_c, dep_d + dep_c),
        ),
        f"{'=' * 80}\n",
    ])
    print(table, flush=True)


# =========================================================================== #
#                        进程入口
# =========================================================================== #

def test_loop(
    local_rank: int,
    num_local_ranks: int,
    args: argparse.Namespace,
) -> None:
    """每个进程的主入口。"""
    # 根据命令行参数设置 DEEPEP_SHMEM_ENABLE 环境变量
    os.environ["DEEPEP_SHMEM_ENABLE"] = str(args.shmem)

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank + 42)

    shmem_status = "enabled" if args.shmem == 1 else "disabled"
    print(
        f"[Rank {rank} | Local rank {local_rank}] Initializing deep_ep_cpp.Buffer... "
        f"(shmem {shmem_status})",
        flush=True,
    )

    # 获取 HCCL 通信组名称（与 deep_ep.Buffer 相同逻辑）
    try:
        pg = group._get_backend(torch.device("npu"))
        moe_all_to_all_group_name = pg.get_hccl_comm_name(rank)
    except Exception:
        moe_all_to_all_group_name = ""

    buffer = deep_ep_cpp.Buffer(
        rank,                          # rank
        num_ranks,                     # num_ranks
        int(2e9),                      # num_nvl_bytes
        0,                             # num_rdma_bytes
        False,                         # low_latency_mode
        moe_all_to_all_group_name,     # moe_all_to_all_group_name
    )
    print(f"[Rank {rank}] Buffer created OK.", flush=True)

    num_tokens = random.randint(1, args.num_tokens)
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    mode = args.mode
    use_quant = args.use_quant

    print_test_config(rank, num_tokens, hidden, num_topk, num_experts, num_ranks)
    if rank == 0:
        print(f"  use_quant    = {use_quant}", flush=True)
    dist.barrier()
    time.sleep(1)
    for _ in range(1000):
        if mode == "fixed":
            test_fixed_correctness(
                buffer, num_tokens, hidden, num_topk, num_experts, num_ranks, rank, group,
                use_quant=use_quant,
            )
        elif mode == "random":
            test_random_correctness(
                buffer, num_tokens, hidden, num_topk, num_experts, num_ranks, rank, group,
                use_quant=use_quant,
            )
        elif mode == "bench":
            bench_performance(
                buffer, num_tokens, hidden, num_topk, num_experts,
                num_ranks, rank, group,
                num_warmups=10,
                num_tests=100,
                use_quant=use_quant,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from: fixed, random, bench")

    dist.barrier()
    dist.destroy_process_group()


# =========================================================================== #
#                           入口
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test deep_ep_cpp.Buffer intranode dispatch/combine directly"
    )
    parser.add_argument(
        "--num-processes", type=int, default=16,
        help="Number of processes to spawn (default: 16)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=1024,
        help="Number of tokens per rank (default: 1024)",
    )
    parser.add_argument(
        "--hidden", type=int, default=7168,
        help="Hidden dimension size (default: 7168)",
    )
    parser.add_argument(
        "--num-topk", type=int, default=8,
        help="Number of top-k experts (default: 8)",
    )
    parser.add_argument(
        "--num-experts", type=int, default=256,
        help="Number of experts (default: 256)",
    )
    parser.add_argument(
        "--mode", type=str, default="random",
        choices=["fixed", "random", "bench"],
        help="Test mode: 'fixed' for fixed-input correctness, "
             "'random' for random-input correctness (default), "
             "'bench' for performance benchmark",
    )
    parser.add_argument(
        "--shmem", type=int, default=1, choices=[0, 1],
        help="Set DEEPEP_SHMEM_ENABLE: 1 to enable shmem, 0 to disable (default: 1)",
    )
    parser.add_argument(
        "--use-quant", action="store_true", default=False,
        help="Enable int8 quantization for dispatch (default: disabled)",
    )
    args = parser.parse_args()

    torch.multiprocessing.spawn(
        fn=test_loop,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
