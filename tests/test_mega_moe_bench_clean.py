"""Clean bench (no profiler_buffer kwarg). Mirrors test_mega_moe_profile.py timing
methodology so numbers are directly comparable."""
import argparse
import os
import random
import time
from typing import Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist


def cast_grouped_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = bf16_weights.shape
    w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
    w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
    w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
    return w, w_sf


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens or num_max_tokens_per_rank
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
    )

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2_weights = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)

    x = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    l1_weights = cast_grouped_weights_to_fp4(l1_weights)
    l2_weights = cast_grouped_weights_to_fp4(l2_weights)
    transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    buffer.x[:num_tokens].copy_(x[0])
    buffer.x_sf[:num_tokens].copy_(x[1])
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)

    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def launch():
        buffer.x[:num_tokens].copy_(x[0])
        buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        deep_gemm.fp8_fp4_mega_moe(
            y,
            transformed_l1_weights, transformed_l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=None,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    dist.barrier()

    def bench(n_iter):
        torch.cuda.synchronize()
        dist.barrier()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iter):
            launch()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n_iter * 1e3

    rounds = args.bench_rounds
    n_iter = args.bench_iters
    samples = sorted(bench(n_iter) for _ in range(rounds))
    median_us = samples[rounds // 2]
    dist_print(f' > rank {rank_idx}: clean kernel median = {median_us:7.1f}us  '
               f'(samples sorted: {[f"{s:.1f}" for s in samples]})',
               once_in_node=False)

    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=512)
    parser.add_argument('--num-tokens', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--intermediate-hidden', type=int, default=3072)
    parser.add_argument('--num-experts', type=int, default=384)
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--activation-clamp', type=float, default=10)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--bench-iters', type=int, default=20)
    parser.add_argument('--bench-rounds', type=int, default=5)
    args = parser.parse_args()

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
