import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


# TODO: skip the test for SM90
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
        if args.num_tokens == 0 else args.num_tokens
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden
    )

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, topk_idx, topk_weights, l1_weights, l2_weights, transformed_l1_weights, transformed_l2_weights
        global cumulative_local_expert_recv_stats_fused
        global cumulative_local_expert_recv_stats_baseline
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        # Check SF requirements
        assert hidden % 128 == 0
        assert intermediate_hidden % 128 == 0
        assert l1_weights.shape[2] % 128 == 0 and l2_weights.shape[2] % 128 == 0

        # Cast inputs to FP8 with per-32 UE8M0 SF
        x = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

        # Cast grouped BF16 weights to FP4 with MN-major SF
        # TODO: merge with `cast_fp8_fp4_with_major`
        def cast_grouped_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            num_groups, n, k = bf16_weights.shape
            w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
            w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
            for i in range(num_groups):
                w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
            w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
            return w, w_sf

        l1_weights = cast_grouped_weights_to_fp4(l1_weights)
        l2_weights = cast_grouped_weights_to_fp4(l2_weights)
        transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    # Profiler buffer layout (all uint64):
    #   [0 .. kNumSMs*4)          kernel span: [clk_s, gt_s, clk_e, gt_e] per SM
    #   [kNumSMs*4 .. kNumSMs*8)  compute span: [clk_s, gt_s, clk_e, gt_e] per SM (leader CTA only)
    #   [kNumSMs*8 .. kNumSMs*10) block counts: [n_l1, n_l2] per SM
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    prof_buf = torch.zeros(num_sms * 10, dtype=torch.int64, device='cuda')

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def run_fused(profiler_buf=None):
        buffer.x[:num_tokens].copy_(x[0])
        buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        # noinspection PyTypeChecker
        deep_gemm.fp8_fp4_mega_moe(
            y,
            transformed_l1_weights, transformed_l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            profiler_buffer=profiler_buf,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            enable_pull=bool(args.enable_pull),
            enable_combine=bool(args.enable_combine)
        )
        return y, cumulative_local_expert_recv_stats_fused

    dist_print('Config:', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # NVML host-side monitoring path: runs the four experiments (violation
    # scatter, polled clock/throttle, polled power, locked-clock compare) and
    # writes per-rank CSVs. Skips the baseline/correctness/kineto paths since
    # they are not needed for NVML reads.
    if args.nvml_monitor:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
        try:
            from nvml_monitor import NvmlMonitorConfig, run_nvml_monitor, NvmlUnavailable
        except Exception as _ex:
            dist_print(f'NVML monitoring requested but helper failed to import: {_ex}',
                       once_in_node=True)
            dist.barrier()
            buffer.destroy()
            dist.destroy_process_group()
            return

        # Parse the batch-size sweep. Empty -> behave like a single-batch run at
        # the current num_max_tokens_per_rank.
        if args.nvml_batch_sizes.strip():
            sweep_batch_sizes = [int(b.strip()) for b in args.nvml_batch_sizes.split(',') if b.strip()]
        else:
            sweep_batch_sizes = [num_max_tokens_per_rank]

        too_big = [b for b in sweep_batch_sizes if b > num_max_tokens_per_rank]
        if too_big:
            dist_print(
                f'--nvml-batch-sizes contains {too_big} which exceed '
                f'--num-max-tokens-per-rank ({num_max_tokens_per_rank}); the symmetric '
                'buffer is sized for the latter, so raise --num-max-tokens-per-rank to '
                f'the largest entry in the sweep ({max(sweep_batch_sizes)}).',
                once_in_node=True)
            dist.barrier()
            buffer.destroy()
            dist.destroy_process_group()
            return

        out_dir = args.nvml_out_dir or os.path.join(os.getcwd(), 'nvml_out')
        if rank_idx == 0:
            os.makedirs(out_dir, exist_ok=True)
        dist.barrier()

        for bs in sweep_batch_sizes:
            # Reassign the closure-captured num_tokens so create_inputs() and
            # run_fused() see the new token count without restructuring.
            num_tokens = bs
            batch_out_dir = os.path.join(out_dir, f'b{bs}')
            if rank_idx == 0:
                os.makedirs(batch_out_dir, exist_ok=True)
            dist.barrier()

            create_inputs()
            for _ in range(5):
                run_fused()
            torch.cuda.synchronize()
            dist.barrier()

            cfg = NvmlMonitorConfig(
                out_dir=batch_out_dir,
                n_iter_violation=args.nvml_iters,
                n_iter_polled=args.nvml_poll_iters,
                n_iter_locked=args.nvml_locked_iters,
                poll_interval_s=max(0.0001, args.nvml_poll_interval_ms / 1000.0),
                locked_mhz=args.nvml_lock_mhz,
                cross_check_with_profiler=not args.nvml_no_cross_check,
            )
            try:
                paths = run_nvml_monitor(
                    config=cfg,
                    rank_idx=rank_idx,
                    local_rank=local_rank,
                    is_distributed_leader=(rank_idx == 0),
                    distributed_barrier=dist.barrier,
                    run_fused=run_fused,
                    sync=torch.cuda.synchronize,
                    prof_buf=prof_buf,
                    num_sms=num_sms,
                    log=lambda m, _bs=bs: dist_print(f'[bs={_bs}] {m}', once_in_node=False),
                )
                dist_print(f' > rank {rank_idx} bs={bs}: NVML outputs: {paths}', once_in_node=False)
            except NvmlUnavailable as _ex:
                dist_print(f' > rank {rank_idx} bs={bs}: NVML monitor skipped: {_ex}',
                           once_in_node=False)

        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = import_baseline()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    ep_buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
        num_topk=num_topk, use_fp8_dispatch=True,
        explicitly_destroy=True,
        allow_multiple_reduction=False,
        gpu_timeout_secs=10, cpu_timeout_secs=30
    ) if is_legacy_loaded else None

    def run_baseline():
        recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
            num_experts=num_experts, expert_alignment=alignment,
            do_cpu_sync=False, do_handle_copy=False,
            do_expand=True, use_tma_aligned_col_major_sf=True,
        )
        n = recv_x[0].size(0)
        l1_y = torch.empty((n, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True, recipe=(1, 1, 32))
        # noinspection PyCallingNonCallable
        l1_y = tilelang_ops.swiglu_apply_weight_to_fp8(
            x=l1_y,
            topk_weights=recv_topk_weights,
            avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
            num_per_channels=32,
            use_col_major_scales=True,
            round_scale=True,
            ue8m0_scale=True,
            output_bf16=False,
            clamp_value=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        l2_y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True, recipe=(1, 1, 32))
        return ep_buffer.combine(l2_y, handle=handle)[0], cumulative_local_expert_recv_stats_baseline

    # Check correctness (must be bitwise identical)
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    # noinspection PyBroadException
    if is_legacy_loaded and num_correctness_tests > 0:
        dist_print('Running correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            for fused_result, baseline_result in zip(run_fused(), run_baseline()):
                assert torch.equal(fused_result, baseline_result)
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
    else:
        create_inputs()

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark — optionally run multiple times and trim worst 20% (env: DG_NUM_BENCH_RUNS)
    _num_bench_runs = int(os.environ.get('DG_NUM_BENCH_RUNS', '1'))
    _bench_args = dict(
        kernel_names='mega_moe',
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.barrier(),
        trace_path=None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json')
    _samples = sorted(bench_kineto(run_fused, **_bench_args) for _ in range(_num_bench_runs))
    if int(os.environ.get('DG_BENCH_BEST', '0')):
        t_fused = _samples[0]
    else:
        _n_keep = max(1, len(_samples) - int(round(len(_samples) * 0.2)))
        t_fused = sum(_samples[:_n_keep]) / _n_keep
    t_baseline = tilelang_bench(run_baseline, _n_warmup=5, _n_repeat=1, backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded else 0

    # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)

    # HBM bytes: weights (FP4 packed = 0.5 bytes) + activations (FP8 = 1 byte) + output (BF16 = 2 bytes)
    num_touched_experts = torch.unique(gathered_topk_idx.flatten()).numel() - 1 # NOTES minus 1 to exclude "-1"
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden // 2 +   # L1 weights (FP4)
        num_touched_experts * hidden * intermediate_hidden // 2 +       # L2 weights (FP4)
        num_recv_tokens * hidden +                                      # L1 acts read (FP8)
        num_recv_tokens * intermediate_hidden +                         # L1 output write (FP8)
        num_recv_tokens * intermediate_hidden +                         # L2 acts read (FP8)
        num_recv_tokens * hidden * 2                                    # L2 output write (BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    approx_factor = t_fused / (t_fused - t_reduction)
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_fused * 1e6:4.0f} us, '
               f'reduction: {t_reduction * 1e6:4.1f} us | '
               f'{safe_div(t_baseline, t_fused):.2f}x legacy')

    # Per-SM profiling: timing + block counts + TFLOPS
    import numpy as np, statistics

    # Replicate BLOCK_M heuristic from heuristics/mega_moe.hpp
    expected_tpe = float(num_tokens) * num_ranks * num_topk / num_experts
    if expected_tpe <= 8.5:
        block_m = 16
    elif expected_tpe <= 16.5:
        block_m = 32
    elif expected_tpe <= 32.5:
        block_m = 64
    elif expected_tpe <= 64.5:
        block_m = 96
    elif expected_tpe <= 96.5:
        block_m = 128
    else:
        block_m = 192

    BLOCK_N = 128
    L1_SHAPE_K = hidden
    L2_SHAPE_K = intermediate_hidden
    flops_per_l1_block = 2 * block_m * BLOCK_N * L1_SHAPE_K
    flops_per_l2_block = 2 * block_m * BLOCK_N * L2_SHAPE_K
    PEAK_OPS_PER_CYCLE_PER_SM = 16384  # mxf8f6f4 UMMA on B200

    n_prof_samples = 5
    for sample_i in range(n_prof_samples):
        prof_buf.zero_()
        run_fused(profiler_buf=prof_buf)
        torch.cuda.synchronize()
        raw = prof_buf.cpu().numpy().view(np.uint64)

        kern_timing = raw[:num_sms * 4].reshape(num_sms, 4)          # kernel span
        comp_timing = raw[num_sms * 4:num_sms * 8].reshape(num_sms, 4)  # compute span
        blocks = raw[num_sms * 8:num_sms * 10].reshape(num_sms, 2)   # block counts

        per_sm_mhz, per_sm_wall_us, per_sm_tflops, per_sm_n_l1, per_sm_n_l2 = [], [], [], [], []
        valid_gt_start, valid_gt_end = [], []
        total_flops = 0

        # Compute span (narrower: dispatch-barrier to MMA-end, leader CTA only)
        comp_per_sm_mhz, comp_per_sm_wall_us, comp_per_sm_tflops = [], [], []
        comp_gt_start, comp_gt_end = [], []
        comp_total_flops = 0

        for sm in range(num_sms):
            nl1, nl2 = int(blocks[sm, 0]), int(blocks[sm, 1])
            per_sm_n_l1.append(nl1)
            per_sm_n_l2.append(nl2)
            sm_flops = nl1 * flops_per_l1_block + nl2 * flops_per_l2_block

            # Kernel span
            cs, gs, ce, ge = int(kern_timing[sm, 0]), int(kern_timing[sm, 1]), int(kern_timing[sm, 2]), int(kern_timing[sm, 3])
            if ce > cs and ge > gs:
                mhz = (ce - cs) * 1000.0 / (ge - gs)
                wall_us = (ge - gs) / 1000.0
                sm_tflops = sm_flops / ((ge - gs) * 1e-9) / 1e12 if (ge - gs) > 0 else 0.0
                per_sm_mhz.append(mhz)
                per_sm_wall_us.append(wall_us)
                per_sm_tflops.append(sm_tflops)
                valid_gt_start.append(gs)
                valid_gt_end.append(ge)
                total_flops += sm_flops

            # Compute span (only leader CTA writes these, so half the SMs will be zero)
            ccs, cgs, cce, cge = int(comp_timing[sm, 0]), int(comp_timing[sm, 1]), int(comp_timing[sm, 2]), int(comp_timing[sm, 3])
            if cce > ccs and cge > cgs:
                c_mhz = (cce - ccs) * 1000.0 / (cge - cgs)
                c_wall_us = (cge - cgs) / 1000.0
                c_sm_tflops = sm_flops / ((cge - cgs) * 1e-9) / 1e12 if (cge - cgs) > 0 else 0.0
                comp_per_sm_mhz.append(c_mhz)
                comp_per_sm_wall_us.append(c_wall_us)
                comp_per_sm_tflops.append(c_sm_tflops)
                comp_gt_start.append(cgs)
                comp_gt_end.append(cge)
                comp_total_flops += sm_flops

        if not per_sm_mhz:
            continue

        overall_wall_ns = max(valid_gt_end) - min(valid_gt_start)
        overall_tflops = total_flops / (overall_wall_ns * 1e-9) / 1e12 if overall_wall_ns > 0 else 0.0
        med_mhz = statistics.median(per_sm_mhz)
        peak_tflops_per_sm = PEAK_OPS_PER_CYCLE_PER_SM * med_mhz * 1e6 / 1e12
        overall_peak_tflops = peak_tflops_per_sm * num_sms

        med_sm_util = statistics.median(per_sm_tflops) / peak_tflops_per_sm * 100 if peak_tflops_per_sm > 0 else 0.0
        overall_util = overall_tflops / overall_peak_tflops * 100 if overall_peak_tflops > 0 else 0.0

        if sample_i == 0:
            dist_print(f' > EP: {rank_idx:2}/{num_ranks} | BLOCK_M={block_m}  num_sms={num_sms}  '
                       f'flops/l1_blk={flops_per_l1_block/1e6:.1f}M  flops/l2_blk={flops_per_l2_block/1e6:.1f}M  '
                       f'total_flops={total_flops/1e9:.1f}G',
                       once_in_node=True)
        dist_print(f' > EP: {rank_idx:2}/{num_ranks} | sample {sample_i} | '
                   f'KERNEL MHz: med={med_mhz:.0f} min={min(per_sm_mhz):.0f} max={max(per_sm_mhz):.0f} | '
                   f'wall_us: med={statistics.median(per_sm_wall_us):.0f} min={min(per_sm_wall_us):.0f} max={max(per_sm_wall_us):.0f} | '
                   f'per-SM TFLOPS: med={statistics.median(per_sm_tflops):.2f} min={min(per_sm_tflops):.2f} max={max(per_sm_tflops):.2f} '
                   f'(peak@med_clk={peak_tflops_per_sm:.1f} util={med_sm_util:.1f}%) | '
                   f'overall={overall_tflops:.0f}/{overall_peak_tflops:.0f} TFLOPS ({overall_util:.1f}%) (gt_span={overall_wall_ns/1e3:.0f}us) | '
                   f'L1_blks: {sum(per_sm_n_l1)} (min={min(per_sm_n_l1)} max={max(per_sm_n_l1)}) '
                   f'L2_blks: {sum(per_sm_n_l2)} (min={min(per_sm_n_l2)} max={max(per_sm_n_l2)})')
        if comp_per_sm_mhz:
            comp_overall_wall_ns = max(comp_gt_end) - min(comp_gt_start)
            comp_overall_tflops = total_flops / (comp_overall_wall_ns * 1e-9) / 1e12 if comp_overall_wall_ns > 0 else 0.0
            comp_med_mhz = statistics.median(comp_per_sm_mhz)
            comp_peak = PEAK_OPS_PER_CYCLE_PER_SM * comp_med_mhz * 1e6 / 1e12
            comp_overall_peak = comp_peak * num_sms
            comp_med_util = statistics.median(comp_per_sm_tflops) / comp_peak * 100 if comp_peak > 0 else 0.0
            comp_overall_util = comp_overall_tflops / comp_overall_peak * 100 if comp_overall_peak > 0 else 0.0
            dist_print(f' > EP: {rank_idx:2}/{num_ranks} | sample {sample_i} | '
                       f'COMPUTE MHz: med={comp_med_mhz:.0f} min={min(comp_per_sm_mhz):.0f} max={max(comp_per_sm_mhz):.0f} | '
                       f'wall_us: med={statistics.median(comp_per_sm_wall_us):.0f} min={min(comp_per_sm_wall_us):.0f} max={max(comp_per_sm_wall_us):.0f} | '
                       f'per-SM TFLOPS: med={statistics.median(comp_per_sm_tflops):.2f} min={min(comp_per_sm_tflops):.2f} max={max(comp_per_sm_tflops):.2f} '
                       f'(peak@med_clk={comp_peak:.1f} util={comp_med_util:.1f}%) | '
                       f'overall={comp_overall_tflops:.0f}/{comp_overall_peak:.0f} TFLOPS ({comp_overall_util:.1f}%) (gt_span={comp_overall_wall_ns/1e3:.0f}us) | '
                       f'n_leader_sms={len(comp_per_sm_mhz)}')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument('--enable-pull', type=int, default=1, help='Enable NVLink pull (0=off, 1=on)')
    parser.add_argument('--enable-combine', type=int, default=1, help='Enable NVLink combine push (0=off, 1=on)')

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')

    # NVML host-side monitoring (off by default; enables the four-experiment flow)
    parser.add_argument('--nvml-monitor', action='store_true',
                        help='Enable NVML host-side monitoring (violation snapshots, polled clock/throttle/power, optional locked-clock compare)')
    parser.add_argument('--nvml-out-dir', type=str, default='',
                        help='Output directory for per-rank NVML CSV files (default: ./nvml_out)')
    parser.add_argument('--nvml-batch-sizes', type=str, default='',
                        help='Comma-separated token counts to sweep for NVML monitoring '
                             '(e.g. "512,8192,32768"). Each value must be <= --num-max-tokens-per-rank. '
                             'When empty, the sweep collapses to a single batch at --num-max-tokens-per-rank.')
    parser.add_argument('--nvml-iters', type=int, default=20,
                        help='Iterations for the violation-snapshot loop')
    parser.add_argument('--nvml-poll-iters', type=int, default=40,
                        help='Iterations for the background polling window (throttle + power)')
    parser.add_argument('--nvml-locked-iters', type=int, default=20,
                        help='Iterations per side of the locked-clock comparison')
    parser.add_argument('--nvml-poll-interval-ms', type=float, default=1.0,
                        help='Target cadence of the background poller in milliseconds')
    parser.add_argument('--nvml-lock-mhz', type=int, default=0,
                        help='If >0, run the locked-clock comparison via `nvidia-smi -lgc {mhz},{mhz}`')
    parser.add_argument('--nvml-no-cross-check', action='store_true',
                        help='Disable kernel-internal MHz cross-check (pure NVML mode)')

    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
