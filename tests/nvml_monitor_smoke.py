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
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        print('CUDA unavailable; cannot run the smoke test', file=sys.stderr)
        sys.exit(2)
    local_rank = 0
    torch.cuda.set_device(local_rank)

    # A no-op stand-in for the Mega MoE kernel: simulates ~200us of GPU work.
    x = torch.empty(1024 * 1024, device=f'cuda:{local_rank}')

    def run_fused(profiler_buf=None):
        x.normal_()
        if profiler_buf is not None:
            # The real kernel writes per-SM timing into prof_buf. The smoke run
            # just zeroes it so the cross-check parser sees no usable rows.
            profiler_buf.zero_()

    num_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count
    prof_buf = torch.zeros(num_sms * 10, dtype=torch.int64, device=f'cuda:{local_rank}')

    cfg = NvmlMonitorConfig(
        out_dir=args.out_dir,
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
        log=print,
    )
    elapsed = time.monotonic() - t0
    print(f'\n[smoke] driver returned in {elapsed:.2f}s')
    print('[smoke] csv outputs:')
    for label, path in paths.items():
        sz = os.path.getsize(path)
        with open(path) as f:
            head = f.readline().rstrip()
            data_lines = sum(1 for _ in f)
        print(f'  {label:18s} {sz:6d} B  header={head!r}  data_rows={data_lines}  ->  {path}')


if __name__ == '__main__':
    main()
