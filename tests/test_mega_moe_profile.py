"""Minimal smoke test: launch mega_moe with sm-profiler buffer attached and dump trace.

Usage:
    torchrun --nproc-per-node=8 tests/test_mega_moe_profile.py
"""
import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist

# sm-profiler removed — kernel now self-instruments via SMEM (see prof_events_smem in
# sm100_fp8_fp4_mega_moe.cuh). Each launch overwrites the first 128 uint64 of profiler_buffer.


# Perfetto colors slices by hashing the slice **name** — `cname` is ignored.
# So to give L1 / L2 distinct uniform colors, the slice name must be uniform per type.
# Wave index is stashed into `args.wave` instead of baked into the name.
EVENT_SCHEME = {
    'dispatch_sync':         {},
    'pull_done':             {},
    'kernel_start':          {},
    'compute_start':         {},
    'compute_end':           {},
    'nvlink_barrier_end':    {},
    'combine_start':         {},
    'combine_end':           {},
    'kernel_end':            {},
}


_NAME_ID_RE = re.compile(r'^(?P<name>.+)_(?P<eid>\d+)$')


def post_process_trace(in_path: str, out_path: str, num_stages: int = 0):
    """Light-format passthrough + optional per-stage pipeline range synthesis.

    The new sm-profiler light exporter writes clean names (`tma_a_issue`, no
    `_<id>` suffix) and stores `event_id` in `args`. We don't rename anything;
    we only synthesize cross-warp ranges when `num_stages > 0`.

    Pairing rule:
      - N-th `tma_a_issue` (load-A warp) ↔ N-th `tma_a_done` (MMA warp)
        => `tma_a_load` range, stage = N % num_stages
      - N-th `mma_issue`   (MMA warp)    ↔ N-th `mma_done`   (load-A warp)
        => `mma` range, stage = N % num_stages

    Stage tracks: pid = source block id (so they group under the same SM),
    tid = 1000 + stage. Both `tma_a_load` and `mma` for stage N share a track,
    showing the buffer-reuse history for that physical pipeline stage.
    """
    with open(in_path) as f:
        doc = json.load(f)
    events = list(doc['traceEvents'])

    if num_stages <= 0:
        with open(out_path, 'w') as f:
            json.dump(doc, f)
        return

    # Group instant events by name and sort by their kernel-side event_id.
    PAIR_NAMES = ('tma_a_issue', 'tma_a_done', 'mma_issue', 'mma_done')
    by_name = defaultdict(list)
    for e in events:
        if e.get('ph') == 'i' and e.get('name') in PAIR_NAMES:
            by_name[e['name']].append(e)
    for nm in by_name:
        by_name[nm].sort(key=lambda x: x.get('args', {}).get('event_id', 0))

    base_pid = next((e['pid'] for e in events
                     if e.get('ph') == 'i' and e.get('name') in PAIR_NAMES), 0)

    def synth(start_evs, end_evs, slice_name, end_offset=0):
        """Pair start[i] with end[i + end_offset]."""
        n = min(len(start_evs), max(0, len(end_evs) - end_offset))
        for i in range(n):
            s_ts = start_evs[i]['ts']
            e_ts = end_evs[i + end_offset]['ts']
            stage = i % num_stages
            events.append({
                'name': slice_name,
                'ph': 'X',
                'ts': s_ts,
                'dur': max(0.0, e_ts - s_ts),
                'pid': base_pid,
                'tid': 1000 + stage,
                'cat': 'pipeline',
                'args': {'iter': i, 'stage': stage},
            })

    # tma_a_issue (load-A) and tma_a_done (MMA) align 1:1 — TMA load completion
    # in MMA's full_barrier wait is the direct counterpart of load-A's TMA issue.
    synth(by_name['tma_a_issue'], by_name['tma_a_done'], 'tma_a_load')
    # mma_issue (MMA) and mma_done (load-A) are offset by num_stages — load-A's
    # first num_stages empty_barrier waits return from initial pre-arrived state
    # without any MMA running. The k-th MMA's empty_barrier.arrive is observed by
    # load-A's (k + num_stages)-th wait return.
    synth(by_name['mma_issue'], by_name['mma_done'], 'mma', end_offset=num_stages)

    # Per-stage thread name metadata for Perfetto.
    for stage in range(num_stages):
        events.append({
            'name': 'thread_name', 'ph': 'M',
            'pid': base_pid, 'tid': 1000 + stage,
            'args': {'name': f'stage_{stage}'},
        })

    doc['traceEvents'] = events
    with open(out_path, 'w') as f:
        json.dump(doc, f)


def merge_rank_traces(rank_paths, out_path, sync_event_name='dispatch_sync'):
    """Merge per-rank traces into a single timeline aligned at `sync_event_name`.

    Each rank's clock64() time origin differs (different SM, slightly different
    boot time). The kernel emits `dispatch_sync` once per dispatch warp right
    after the cross-rank `nvlink_barrier` returns, so all ranks pass through it
    at the same wall-clock instant. We treat rank-0's first `dispatch_sync` as
    t=0 and shift each other rank's timeline so their first `dispatch_sync`
    lands on the same t.

    Each rank gets its own pid (= rank index) so Perfetto shows them as separate
    process groups; tids and the synthesized stage tracks (1000+stage) inside
    each rank are preserved unchanged.
    """
    docs = []
    for p in rank_paths:
        with open(p) as f:
            docs.append(json.load(f))

    def first_sync_ts(events):
        for e in events:
            if e.get('ph') == 'i' and e.get('name') == sync_event_name:
                return e['ts']
        return None

    sync_ts = [first_sync_ts(d['traceEvents']) for d in docs]
    missing = [i for i, t in enumerate(sync_ts) if t is None]
    if missing:
        raise RuntimeError(f'rank(s) {missing} have no `{sync_event_name}` event — cannot align')

    # Step 1: per-rank shift so each rank's `sync_event` lands at the SAME ts.
    # Step 2: a global second shift so the earliest event lands at t=0 (Perfetto
    #         hides events with negative ts). Both shifts are uniform across
    #         ranks → cross-rank alignment is preserved.
    rank_shifted = []
    for rank_idx, (doc, ts0) in enumerate(zip(docs, sync_ts)):
        delta = -ts0  # makes this rank's sync land at 0 (relative)
        for e in doc['traceEvents']:
            e2 = dict(e)
            e2['pid'] = rank_idx
            if 'ts' in e2:
                e2['ts'] = e2['ts'] + delta
            rank_shifted.append(e2)

    # Find the earliest non-meta event (only 'X', 'i', 'B' have meaningful ts)
    candidate_ts = [e['ts'] for e in rank_shifted
                    if 'ts' in e and e.get('ph') in ('X', 'i', 'B')]
    global_shift = -min(candidate_ts) if candidate_ts and min(candidate_ts) < 0 else 0
    merged = []
    for e in rank_shifted:
        if global_shift and 'ts' in e:
            e['ts'] = e['ts'] + global_shift
        merged.append(e)
        # Process name metadata
        merged.append({
            'name': 'process_name', 'ph': 'M',
            'pid': rank_idx, 'tid': 0,
            'args': {'name': f'rank_{rank_idx}'},
        })

    with open(out_path, 'w') as f:
        json.dump({'traceEvents': merged}, f)


# Profiler event ids — must match the constants in sm100_fp8_fp4_mega_moe.cuh
EVT_DISPATCH_SYNC         = 0
EVT_PULL_DONE             = 1
EVT_COMPUTE_START         = 2
EVT_COMPUTE_END           = 3
EVT_NVLINK_BARRIER_END    = 4
EVT_COMBINE_END           = 5
EVT_KERNEL_END            = 6
EVT_KERNEL_START          = 7
EVT_COMBINE_START         = 8


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
    # Deterministic per-rank seeding: same `--seed` => identical topk distribution
    # across runs. `torch.manual_seed` also seeds all CUDA generators.
    seed = args.seed * 100003 + rank_idx
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens or num_max_tokens_per_rank
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
    )

    # Inputs
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

    # Self-instrumented profiler buffer: 16 warps × 8 events × 8 bytes = 1024 bytes.
    # Block 0 dumps its SMEM region into the first 128 uint64 entries on every launch.
    # Cycle-to-us conversion uses --actual-sm-clock-mhz (defaults to current nvidia-smi reading).
    prof_buf = torch.zeros(132, dtype=torch.int64, device='cuda')  # 128 events + 4 effective-MHz meta entries
    sm_clock_mhz_for_parse = args.actual_sm_clock_mhz if args.actual_sm_clock_mhz > 0 else sm_clock_mhz
    if sm_clock_mhz_for_parse <= 0:
        sm_clock_mhz_for_parse = 1965  # fallback
    dist_print(f' > prof_buf {prof_buf.numel() * prof_buf.element_size()} B '
               f'(SMEM-self-instrumented, sm_clock for parse: {sm_clock_mhz_for_parse} MHz)',
               once_in_node=True)
    EVENT_ID_TO_NAME = {
        EVT_DISPATCH_SYNC:        'dispatch_sync',
        EVT_PULL_DONE:            'pull_done',
        EVT_COMPUTE_START:        'compute_start',
        EVT_COMPUTE_END:          'compute_end',
        EVT_NVLINK_BARRIER_END:   'nvlink_barrier_end',
        EVT_COMBINE_END:          'combine_end',
        EVT_KERNEL_END:           'kernel_end',
        EVT_KERNEL_START:         'kernel_start',
        EVT_COMBINE_START:        'combine_start',
    }

    # Stage inputs into the symmetric buffer
    buffer.x[:num_tokens].copy_(x[0])
    buffer.x_sf[:num_tokens].copy_(x[1])
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)

    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def launch(profiler_buf):
        # restage inputs (debug mode zeros buffer between calls)
        buffer.x[:num_tokens].copy_(x[0])
        buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        deep_gemm.fp8_fp4_mega_moe(
            y,
            transformed_l1_weights, transformed_l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=None,
            profiler_buffer=profiler_buf,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )

    # Warmup BOTH paths so the first bench() launch isn't a cold launch
    for _ in range(args.warmup):
        launch(None)
    for _ in range(args.warmup):
        launch(prof_buf)
    torch.cuda.synchronize()
    dist.barrier()

    # === Correctness check: y must match between launch(None) and launch(prof_buf) ===
    # The kernel always runs the SMEM-instrumented prof_record; only the final dump
    # differs based on profiler_buffer arg. y should be bit-identical.
    launch(None)
    torch.cuda.synchronize()
    y_clean = y.clone()
    launch(prof_buf)
    torch.cuda.synchronize()
    if not torch.equal(y_clean, y):
        diff_max = (y_clean.float() - y.float()).abs().max().item()
        diff_mean = (y_clean.float() - y.float()).abs().mean().item()
        raise RuntimeError(
            f'rank {rank_idx}: kernel output differs between launch(None) and launch(prof_buf)! '
            f'max abs diff={diff_max:.6f}, mean abs diff={diff_mean:.6f}'
        )
    if not torch.isfinite(y).all().item():
        raise RuntimeError(f'rank {rank_idx}: kernel output contains NaN/Inf')
    # NOTE: under pull-OFF mask the output is legitimately all-zero (no remote tokens).
    # Equality between with/without prof dump is the real correctness signal.
    dist_print(f' > rank {rank_idx}: correctness check passed '
               f'(y identical with vs without prof dump; sum|y|={y.abs().sum().item():.2e})',
               once_in_node=True)

    # A/B benchmark: kernel time with vs without profiler buffer
    def bench(profiler_buf, n_iter):
        # Sync before to flush any prior async work
        torch.cuda.synchronize()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            launch(profiler_buf)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iter * 1e3  # us per iter

    n_iter = args.bench_iters
    rounds = args.bench_rounds
    # Round-0 is the only "clean" measurement (subsequent rounds drift +25-50us).
    off_rounds = [bench(None, n_iter) for _ in range(rounds)]
    on_rounds  = [bench(prof_buf, n_iter) for _ in range(rounds)]
    dist_print(f' > rank {rank_idx}: OFF rounds = ' + ' '.join(f'{t:6.1f}' for t in off_rounds) +
               f'  ON rounds = ' + ' '.join(f'{t:6.1f}' for t in on_rounds) +
               f'  | round0: OFF={off_rounds[0]:.1f}us ON={on_rounds[0]:.1f}us '
               f'overhead=+{on_rounds[0]-off_rounds[0]:.1f}us '
               f'({100*(on_rounds[0]-off_rounds[0])/off_rounds[0]:+.1f}%)',
               once_in_node=False)

    # Always grab per-rank recv counts — needed to compute dispatch NVLink BW.
    recv_stats = torch.zeros(num_experts_per_rank, dtype=torch.int32, device='cuda')
    buffer.x[:num_tokens].copy_(x[0])
    buffer.x_sf[:num_tokens].copy_(x[1])
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)
    deep_gemm.fp8_fp4_mega_moe(
        y, transformed_l1_weights, transformed_l2_weights, buffer,
        cumulative_local_expert_recv_stats=recv_stats,
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math),
    )
    torch.cuda.synchronize()
    counts = recv_stats.cpu().tolist()
    num_recv_tokens = sum(counts)
    if args.show_recv_counts:
        dist_print(f' > rank {rank_idx}: received {num_recv_tokens} tokens '
                   f'(min={min(counts)} max={max(counts)} std={(sum((c-num_recv_tokens/len(counts))**2 for c in counts)/len(counts))**0.5:.1f})',
                   once_in_node=False)

    # SMEM-self-instrumented kernel writes events to prof_buf at each launch (block 0 only).
    # Each launch overwrites the buffer, so we snapshot to host memory between launches.
    import numpy as np
    n_meas = args.bench_iters
    snapshots = []
    for _ in range(n_meas):
        prof_buf.zero_()
        launch(prof_buf)
        torch.cuda.synchronize()
        snapshots.append(prof_buf.cpu().numpy().view(np.uint64).copy())

    # Parse snapshots into by_warp[tid][event_name] = [(launch_idx, ts_us), ...]
    import statistics
    from collections import defaultdict
    by_warp = defaultdict(lambda: defaultdict(list))
    eff_mhz_list = []  # per-launch effective MHz from kernel-internal clock64 / globaltimer.
    for launch_idx, snap in enumerate(snapshots):
        for warp_idx in range(16):
            for slot in range(8):
                entry = int(snap[warp_idx * 8 + slot])
                if entry == 0:
                    continue
                clock = entry >> 8
                eid = int(entry & 0xff)
                if eid not in EVENT_ID_TO_NAME:
                    continue
                ts_us = clock / sm_clock_mhz_for_parse  # cycles / MHz = us
                tid = warp_idx * 32  # mimic sm-profiler tid convention
                by_warp[tid][EVENT_ID_TO_NAME[eid]].append((launch_idx, ts_us))
        # Effective MHz (per launch): meta[128..131] = (cyc_start, ns_start, cyc_end, ns_end)
        cyc_s, ns_s, cyc_e, ns_e = int(snap[128]), int(snap[129]), int(snap[130]), int(snap[131])
        if cyc_e > cyc_s and ns_e > ns_s:
            eff_mhz = (cyc_e - cyc_s) * 1000.0 / (ns_e - ns_s)
            eff_mhz_list.append(eff_mhz)
    # Already sorted by launch_idx (we appended in order).

    if eff_mhz_list:
        eff_med = statistics.median(eff_mhz_list)
        eff_min = min(eff_mhz_list)
        eff_max = max(eff_mhz_list)
        # use a sliding-window-average view too
        srt = sorted(eff_mhz_list)
        n_e = len(srt)
        eff_p25, eff_p75 = srt[n_e // 4], srt[3 * n_e // 4]
        dist_print(
            f' > rank {rank_idx}: effective MHz (kernel-internal): '
            f'median {eff_med:.0f}  min {eff_min:.0f}  p25 {eff_p25:.0f}  '
            f'p75 {eff_p75:.0f}  max {eff_max:.0f}  (n={n_e} launches)',
            once_in_node=False,
        )

    # Optionally write a minimal JSON trace for visualization (Perfetto-compatible).
    run_dir = os.path.join(args.dump_dir, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f'rank{rank_idx}.json')
    trace_events = []
    for tid, names in by_warp.items():
        for name, items in names.items():
            for eid, ts in items:
                trace_events.append({'name': name, 'ph': 'i', 'pid': 0, 'tid': tid, 'ts': ts,
                                     's': 't', 'args': {'event_id': eid}})
    with open(out_path, 'w') as f:
        json.dump({'traceEvents': trace_events, 'displayTimeUnit': 'ns'}, f)
    dist_print(f' > rank {rank_idx}: trace ({len(trace_events)} events) -> {out_path}', once_in_node=False)

    # Per-launch dispatch BW: launch k → sync = min(across 4 dispatch warps) of warp's
    # k-th sync event; done = max of warp's k-th done event.
    dispatch_warps = [tid for tid, d in by_warp.items() if 'dispatch_sync' in d]
    if dispatch_warps:
        n_launches = min(len(by_warp[t]['dispatch_sync']) for t in dispatch_warps)
        per_token_bytes = hidden + (hidden // 128) * 4 + 4
        rank_bytes = num_recv_tokens * per_token_bytes
        pull_us_list = []
        for k in range(n_launches):
            sync_k = min(by_warp[t]['dispatch_sync'][k][1] for t in dispatch_warps)
            done_k = max(by_warp[t]['pull_done'   ][k][1] for t in dispatch_warps)
            pull_us_list.append(done_k - sync_k)
        pull_med = statistics.median(pull_us_list)
        bw = rank_bytes / 1e9 / (pull_med * 1e-6) if pull_med > 0 else 0.0
        dist_print(
            f' > rank {rank_idx}: dispatch pull median over {n_launches} launches '
            f'{pull_med:6.1f} us  (min {min(pull_us_list):.1f}, max {max(pull_us_list):.1f}) '
            f'→ {bw:.0f} GB/s NVLink ({rank_bytes/1e6:.1f} MB)',
            once_in_node=False,
        )

    # True kernel device time: kernel_start = min over ALL warps in block 0; kernel_end = max over all warps.
    # All warps in the same block share clock64, so cross-warp min/max is timestamp-safe.
    ks_warps = [tid for tid, d in by_warp.items() if 'kernel_start' in d]
    ke_warps_all = [tid for tid, d in by_warp.items() if 'kernel_end' in d]
    if ks_warps and ke_warps_all:
        n_ks = min(len(by_warp[t]['kernel_start']) for t in ks_warps)
        n_ke = min(len(by_warp[t]['kernel_end']) for t in ke_warps_all)
        n_launches_kk = min(n_ks, n_ke)
        kernel_us_list = []
        for k in range(n_launches_kk):
            t_start = min(by_warp[t]['kernel_start'][k][1] for t in ks_warps)
            t_end   = max(by_warp[t]['kernel_end'  ][k][1] for t in ke_warps_all)
            kernel_us_list.append(t_end - t_start)
        dist_print(
            f' > rank {rank_idx}: kernel_start->kernel_end median {statistics.median(kernel_us_list):6.1f} us '
            f'(min {min(kernel_us_list):.1f}, max {max(kernel_us_list):.1f}) over {n_launches_kk} launches '
            f'[true device time, min(start) -> max(end) across {len(ks_warps)} warps]',
            once_in_node=False,
        )

        # Per-warp first-in / last-out tids to identify which warp class fires first/last on average.
        first_in_count = {}
        last_out_count = {}
        for k in range(n_launches_kk):
            first_tid = min(ks_warps, key=lambda t: by_warp[t]['kernel_start'][k][1])
            last_tid  = max(ke_warps_all, key=lambda t: by_warp[t]['kernel_end'  ][k][1])
            first_in_count[first_tid] = first_in_count.get(first_tid, 0) + 1
            last_out_count[last_tid]  = last_out_count.get(last_tid,  0) + 1
        dist_print(
            f' > rank {rank_idx}: first-in tids (count): {sorted(first_in_count.items(), key=lambda x:-x[1])[:3]}',
            once_in_node=False,
        )
        dist_print(
            f' > rank {rank_idx}: last-out tids (count): {sorted(last_out_count.items(), key=lambda x:-x[1])[:3]}',
            once_in_node=False,
        )

    # Per-launch kernel_start -> compute_start using load-A warp's events (within-warp safe).
    # Captures: prologue + reg_dealloc + first L1 arrival count wait, on the load-A warp specifically.
    le_warps = [tid for tid, d in by_warp.items() if 'kernel_start' in d and 'compute_start' in d]
    if le_warps:
        tid = le_warps[0]
        n_launches = min(len(by_warp[tid]['kernel_start']), len(by_warp[tid]['compute_start']))
        prologue_us = [by_warp[tid]['compute_start'][k][1] - by_warp[tid]['kernel_start'][k][1] for k in range(n_launches)]
        prologue_med = statistics.median(prologue_us)
        dist_print(
            f' > rank {rank_idx}: kernel_start(load-A)->compute_start median {prologue_med:6.1f} us '
            f'(min {min(prologue_us):.1f}, max {max(prologue_us):.1f}) [load-A prologue + first L1 wait]',
            once_in_node=False,
        )

    # Per-launch compute TFLOPS. compute_start is on load-A warp; compute_end is on epilogue warp 0.
    # Both are in the same exported block (same SM), so clock64 timestamps are comparable.
    cs_warps = [tid for tid, d in by_warp.items() if 'compute_start' in d]
    ce_warps = [tid for tid, d in by_warp.items() if 'compute_end' in d]
    if cs_warps and ce_warps:
        cs = by_warp[cs_warps[0]]['compute_start']
        ce = by_warp[ce_warps[0]]['compute_end']
        n_launches = min(len(cs), len(ce))
        compute_us_list = [ce[k][1] - cs[k][1] for k in range(n_launches)]
        compute_med = statistics.median(compute_us_list)
        rank_flops = 6.0 * num_recv_tokens * hidden * intermediate_hidden
        tflops = rank_flops / 1e12 / (compute_med * 1e-6) if compute_med > 0 else 0.0
        dist_print(
            f' > rank {rank_idx}: compute median over {n_launches} launches '
            f'{compute_med:6.1f} us  (min {min(compute_us_list):.1f}, max {max(compute_us_list):.1f}) '
            f'→ {tflops:.0f} TFLOPS ({rank_flops/1e12:.2f} TFLOP)',
            once_in_node=False,
        )

    # Per-launch kernel_end on dispatch warps (within-warp delta is timestamp-safe).
    # dispatch_sync->kernel_end covers: dispatch pull + workspace cleanup + grid_sync (waits for epilogue) + cross-rank nvlink_barrier.
    de_warps = [tid for tid, d in by_warp.items() if 'kernel_end' in d and 'dispatch_sync' in d]
    if de_warps:
        n_launches = min(len(by_warp[t]['kernel_end']) for t in de_warps)
        ds_to_ke = []
        pd_to_ke = []
        for k in range(n_launches):
            warp_dks = [by_warp[t]['kernel_end'][k][1] - by_warp[t]['dispatch_sync'][k][1] for t in de_warps]
            ds_to_ke.append(max(warp_dks))
            warp_pks = [by_warp[t]['kernel_end'][k][1] - by_warp[t]['pull_done'][k][1] for t in de_warps if 'pull_done' in by_warp[t]]
            if warp_pks:
                pd_to_ke.append(max(warp_pks))
        dist_print(
            f' > rank {rank_idx}: dispatch_sync->kernel_end median {statistics.median(ds_to_ke):6.1f} us '
            f'(min {min(ds_to_ke):.1f}, max {max(ds_to_ke):.1f}) over {n_launches} launches',
            once_in_node=False,
        )
        if pd_to_ke:
            dist_print(
                f' > rank {rank_idx}: pull_done->kernel_end  median {statistics.median(pd_to_ke):6.1f} us '
                f'(min {min(pd_to_ke):.1f}, max {max(pd_to_ke):.1f})  [post-pull cleanup + grid_sync + nvlink_barrier]',
                once_in_node=False,
            )

    # Per-launch chain windows on epilogue warp 0 (compute_end -> nvlink_barrier_end -> combine_start -> combine_end).
    # All four events fire on the same warp (epilogue warp 0), so within-warp deltas are clock-safe.
    ep_warps = [tid for tid, d in by_warp.items()
                if {'compute_end', 'nvlink_barrier_end', 'combine_start', 'combine_end'}.issubset(d.keys())]
    if ep_warps:
        tid = ep_warps[0]
        n_launches = min(len(by_warp[tid][nm]) for nm in ('compute_end', 'nvlink_barrier_end', 'combine_start', 'combine_end'))
        ce_to_ne = [by_warp[tid]['nvlink_barrier_end'][k][1] - by_warp[tid]['compute_end'    ][k][1] for k in range(n_launches)]
        ne_to_cs = [by_warp[tid]['combine_start'     ][k][1] - by_warp[tid]['nvlink_barrier_end'][k][1] for k in range(n_launches)]
        cs_to_ce = [by_warp[tid]['combine_end'       ][k][1] - by_warp[tid]['combine_start'   ][k][1] for k in range(n_launches)]
        dist_print(
            f' > rank {rank_idx}: compute_end->nvlink_barrier_end median {statistics.median(ce_to_ne):5.1f} us '
            f'(min {min(ce_to_ne):.1f}, max {max(ce_to_ne):.1f})  [L2 cross-rank nvlink_barrier]',
            once_in_node=False,
        )
        dist_print(
            f' > rank {rank_idx}: nvlink_barrier_end->combine_start median {statistics.median(ne_to_cs):5.1f} us '
            f'(min {min(ne_to_cs):.1f}, max {max(ne_to_cs):.1f})  [dispatch-with-epilogue sync + setup]',
            once_in_node=False,
        )
        dist_print(
            f' > rank {rank_idx}: combine_start->combine_end median {statistics.median(cs_to_ce):5.1f} us '
            f'(min {min(cs_to_ce):.1f}, max {max(cs_to_ce):.1f})  [combine reduce loop body]',
            once_in_node=False,
        )

        # combine_end (epilogue warp 0) -> kernel_end (dispatch warp). Both are in block 0
        # (same SM, same clock64), so cross-warp delta is timestamp-safe.
        # Covers: epilogue warp tail + dispatch warp's post-pull cleanup wait + final cross-rank kAfterWorkspaceCleanBarrierTag nvlink_barrier.
        ke_warps = [t for t, d in by_warp.items() if 'kernel_end' in d]
        if ke_warps:
            n2 = min(n_launches, min(len(by_warp[t]['kernel_end']) for t in ke_warps))
            ce_to_kend = [max(by_warp[t]['kernel_end'][k][1] for t in ke_warps) - by_warp[tid]['combine_end'][k][1] for k in range(n2)]
            dist_print(
                f' > rank {rank_idx}: combine_end->kernel_end median {statistics.median(ce_to_kend):5.1f} us '
                f'(min {min(ce_to_kend):.1f}, max {max(ce_to_kend):.1f})  [final cross-rank nvlink_barrier (kAfterWorkspaceClean)]',
                once_in_node=False,
            )

    # All ranks must finish writing before rank 0 can merge.
    dist.barrier()

    if rank_idx == 0:
        rank_paths = [os.path.join(run_dir, f'rank{r}.json') for r in range(num_ranks)]
        merged_path = os.path.join(run_dir, 'merged.json')
        try:
            merge_rank_traces(rank_paths, merged_path)
            print(f' > merged trace ({num_ranks} ranks aligned at dispatch_sync) → {merged_path}')
        except RuntimeError as e:
            print(f' > merge skipped: {e}')

    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0,
                        help='deterministic seed for topk routing (same seed => same per-rank token distribution)')
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=512)
    parser.add_argument('--num-tokens', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--intermediate-hidden', type=int, default=3072)
    parser.add_argument('--num-experts', type=int, default=384)
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--activation-clamp', type=float, default=10)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--dump-dir', type=str, default='/tmp/mega_moe_traces')
    parser.add_argument('--run-tag', type=str, default='',
                        help='subdirectory name; defaults to bs{N}_h{H}_e{E}_topk{K}_{timestamp}')
    parser.add_argument('--warmup', type=int, default=10, help='warmup launches before the measured one')
    parser.add_argument('--actual-sm-clock-mhz', type=int, default=0,
                        help='actual SM clock in MHz to use for trace cycle→us conversion (default 0 = use cudaDeviceProp.clock_rate, the max boost)')
    parser.add_argument('--num-profiled-sms', type=int, default=0,
                        help='record events from SMs 0..N-1 (default 0 = all SMs)')
    parser.add_argument('--max-events-per-group', type=int, default=2048,
                        help='per-warp event capacity (mbarrier-wait mode needs ~1500 at bs=1024)')
    parser.add_argument('--export-block-id', type=int, default=0,
                        help='which SM (block) to export to JSON (light traces are 1-block by convention)')
    parser.add_argument('--show-recv-counts', action='store_true',
                        help='launch once with cumulative_local_expert_recv_stats and print the per-expert counts')
    parser.add_argument('--num-stages', type=int, default=0,
                        help='kernel kNumStages; if > 0, post-process pairs tma/mma instants into per-stage ranges')
    parser.add_argument('--bench-iters', type=int, default=20, help='launches per timing round')
    parser.add_argument('--bench-rounds', type=int, default=5, help='timing rounds; median is reported')
    args = parser.parse_args()

    # Capture current SM clock (GPU 0) so trace dirs are self-describing — useful
    # when comparing locked-vs-unlocked clock runs.
    sm_clock_mhz = 0
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader,nounits', '-i', '0'],
            stderr=subprocess.DEVNULL).decode().strip().split('\n')[0]
        sm_clock_mhz = int(out)
    except Exception:
        pass

    if not args.run_tag:
        args.run_tag = (f'bs{args.num_max_tokens_per_rank}_h{args.hidden}'
                        f'_e{args.num_experts}_topk{args.num_topk}'
                        f'_clk{sm_clock_mhz}MHz_'
                        + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(args.dump_dir, exist_ok=True)
    print(f'[host] run_tag = {args.run_tag}  (SM clock at start: {sm_clock_mhz} MHz)')

    # If launched via torchrun (LOCAL_RANK set), run test() directly in this process
    # instead of spawning. This is needed for ncu profiling which doesn't follow mp.spawn children.
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # init_dist treats env WORLD_SIZE as num_nodes and adds local_rank to it,
        # so under torchrun (where WORLD_SIZE = total ranks) we override it.
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        test(local_rank, world_size, args)
    else:
        torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
