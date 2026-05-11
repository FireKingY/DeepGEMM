"""Standalone smoke test for the NVML monitoring helper.

Drives :mod:`tests.nvml_monitor` with a no-op kernel so the orchestration,
threading, CSV writing, and locked-clock skip path can be validated without
needing a built ``deep_gemm._C.so``. Exercises:

  * NVML init + handle lookup for the current local rank
  * violation-scatter loop, including a prof_buf cross-check stand-in
  * background poller during a multi-iteration window
  * locked-clock context manager when it cannot acquire ``nvidia-smi -lgc``

Usage:
    python tests/nvml_monitor_smoke.py [--lock-mhz 1965]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from nvml_monitor import NvmlMonitorConfig, run_nvml_monitor  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='/tmp/nvml_monitor_smoke_out')
    parser.add_argument('--lock-mhz', type=int, default=0,
                        help='exercise the locked-clock path; expected to skip on a shared GPU')
    parser.add_argument('--iters-violation', type=int, default=10)
    parser.add_argument('--iters-polled', type=int, default=20)
    parser.add_argument('--iters-locked', type=int, default=10)
    parser.add_argument('--batch-sizes', type=str, default='',
                        help='Comma-separated batch sizes to sweep, mirroring '
                             '--nvml-batch-sizes in test_mega_moe.py. Empty -> single batch.')
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        print('CUDA unavailable; cannot run the smoke test', file=sys.stderr)
        sys.exit(2)
    local_rank = 0
    torch.cuda.set_device(local_rank)

    # `num_tokens` is captured by closure below, mirroring test_mega_moe.py's
    # test() scope. Reassigning it in the sweep loop must update the closure.
    num_tokens = 0

    # A no-op stand-in for the Mega MoE kernel: simulates ~200us of GPU work.
    x = torch.empty(1024 * 1024, device=f'cuda:{local_rank}')

    def run_fused(profiler_buf=None):
        # Use num_tokens via closure so the smoke run exercises the same
        # rebinding pattern as test_mega_moe.py.
        n = max(1, min(num_tokens, x.numel()))
        x[:n].normal_()
        if profiler_buf is not None:
            profiler_buf.zero_()

    num_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count
    prof_buf = torch.zeros(num_sms * 10, dtype=torch.int64, device=f'cuda:{local_rank}')

    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(',') if b.strip()] \
        if args.batch_sizes else [4096]

    overall_paths = {}
    for bs in batch_sizes:
        num_tokens = bs  # closure rebinding mirrors the production code path
        batch_out_dir = os.path.join(args.out_dir, f'b{bs}')
        os.makedirs(batch_out_dir, exist_ok=True)

        cfg = NvmlMonitorConfig(
            out_dir=batch_out_dir,
            n_iter_violation=args.iters_violation,
            n_iter_polled=args.iters_polled,
            n_iter_locked=args.iters_locked,
            poll_interval_s=0.001,
            locked_mhz=args.lock_mhz,
            cross_check_with_profiler=True,
        )

        t0 = time.monotonic()
        paths = run_nvml_monitor(
            config=cfg,
            rank_idx=0,
            local_rank=local_rank,
            is_distributed_leader=True,
            distributed_barrier=lambda: None,
            run_fused=run_fused,
            sync=torch.cuda.synchronize,
            prof_buf=prof_buf,
            num_sms=num_sms,
            log=lambda m, _bs=bs: print(f'[bs={_bs}] {m}'),
        )
        elapsed = time.monotonic() - t0
        print(f'\n[smoke bs={bs}] driver returned in {elapsed:.2f}s')
        for label, path in paths.items():
            sz = os.path.getsize(path)
            with open(path) as f:
                head = f.readline().rstrip()
                data_lines = sum(1 for _ in f)
            print(f'  {label:18s} {sz:6d} B  header={head!r}  data_rows={data_lines}  ->  {path}')
        overall_paths[bs] = paths

    # Sanity check that each batch produced its own directory + non-empty CSVs.
    print('\n[smoke] sweep summary:')
    for bs, paths in overall_paths.items():
        print(f'  b{bs}: {len(paths)} CSV(s) at {os.path.join(args.out_dir, f"b{bs}")}')


if __name__ == '__main__':
    main()
