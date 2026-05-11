"""Host-side NVML monitoring helpers for the Mega MoE kernel runs.

This module is opt-in: it is only imported when `--nvml-monitor` is passed to
`tests/test_mega_moe.py`. It supplements the in-kernel `clock64()/globaltimer`
profiler with host-visible state that the kernel cannot observe:

  * cumulative power-violation time deltas across each kernel launch,
  * background-thread time series of SM clock, throttle-reason bitmask, and
    instantaneous power,
  * an optional locked-vs-free clock comparison driven by ``nvidia-smi -lgc``.

The helpers are designed to coexist with the kernel-internal profiler so a
single run can produce both data sources for cross-check.
"""

from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Lazy NVML import + clean error messages
# ---------------------------------------------------------------------------


_PYNVML_IMPORT_ERROR: Optional[str] = None
try:
    import pynvml as _pynvml  # nvidia-ml-py
except Exception as ex:  # pragma: no cover - exercised by the reverse test
    _pynvml = None
    _PYNVML_IMPORT_ERROR = (
        f'pynvml is required for NVML monitoring (pip install nvidia-ml-py): {ex}'
    )


class NvmlUnavailable(RuntimeError):
    """Raised when pynvml or the NVML driver cannot be used."""


def _require_pynvml():
    if _pynvml is None:
        raise NvmlUnavailable(_PYNVML_IMPORT_ERROR or 'pynvml not importable')
    return _pynvml


# ---------------------------------------------------------------------------
# Throttle-reason bit table
# ---------------------------------------------------------------------------

# Bit constants are stable across driver versions. Pin the table locally so the
# CSV columns stay deterministic even if pynvml renames a symbol between
# releases.
THROTTLE_REASONS: Tuple[Tuple[str, int], ...] = (
    ('GPU_IDLE',                        0x0000000000000001),
    ('APPLICATIONS_CLOCKS_SETTING',     0x0000000000000002),
    ('SW_POWER_CAP',                    0x0000000000000004),
    ('HW_SLOWDOWN',                     0x0000000000000008),
    ('SYNC_BOOST',                      0x0000000000000010),
    ('SW_THERMAL_SLOWDOWN',             0x0000000000000020),
    ('HW_THERMAL_SLOWDOWN',             0x0000000000000040),
    ('HW_POWER_BRAKE_SLOWDOWN',         0x0000000000000080),
    ('DISPLAY_CLOCK_SETTING',           0x0000000000000100),
)


# ---------------------------------------------------------------------------
# Session: lazy init, handle ownership, context-manager cleanup
# ---------------------------------------------------------------------------


class NvmlSession:
    """Owns ``nvmlInit`` lifetime and a single device handle.

    The device handle is selected by ``device_index``. When ``device_index`` is
    ``None``, the session falls back to index ``0`` so the helper works in a
    plain single-GPU process without ``torch.distributed`` initialised.
    """

    def __init__(self, device_index: Optional[int]):
        nv = _require_pynvml()
        self._nv = nv
        self._initialised = False
        try:
            nv.nvmlInit()
            self._initialised = True
        except nv.NVMLError as ex:
            raise NvmlUnavailable(f'nvmlInit failed: {ex}') from ex

        idx = 0 if device_index is None else int(device_index)
        try:
            self.handle = nv.nvmlDeviceGetHandleByIndex(idx)
        except nv.NVMLError as ex:
            self.shutdown()
            raise NvmlUnavailable(
                f'nvmlDeviceGetHandleByIndex({idx}) failed: {ex}'
            ) from ex
        self.device_index = idx

    def __enter__(self) -> 'NvmlSession':
        return self

    def __exit__(self, *exc):
        self.shutdown()

    def shutdown(self):
        if self._initialised:
            try:
                self._nv.nvmlShutdown()
            except Exception:
                pass
            self._initialised = False

    # ---- read helpers ---------------------------------------------------

    def violation_status_power_ns(self) -> int:
        """Cumulative power-violation time in nanoseconds (monotonic counter)."""
        nv = self._nv
        v = nv.nvmlDeviceGetViolationStatus(self.handle, nv.NVML_PERF_POLICY_POWER)
        return int(v.violationTime)

    def sm_clock_mhz(self) -> int:
        nv = self._nv
        return int(nv.nvmlDeviceGetClockInfo(self.handle, nv.NVML_CLOCK_SM))

    def throttle_reasons_bitmask(self) -> int:
        nv = self._nv
        return int(nv.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle))

    def power_milliwatts(self) -> int:
        nv = self._nv
        return int(nv.nvmlDeviceGetPowerUsage(self.handle))


# ---------------------------------------------------------------------------
# Background polling thread
# ---------------------------------------------------------------------------


@dataclass
class PollSample:
    timestamp_ns: int
    sm_mhz: int
    throttle_bitmask: int
    power_watts: float


@dataclass
class PollingResult:
    samples: List[PollSample] = field(default_factory=list)
    reason_seconds: Dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0
    short_window_warning: Optional[str] = None


class BackgroundPoller:
    """Polls clock / throttle bitmask / power on a background thread.

    Cadence is approximately ``poll_interval_s`` per loop. The thread stops as
    soon as the stop event is set, so it does not deadlock if the foreground
    kernel returns earlier than expected.
    """

    def __init__(self, session: NvmlSession, poll_interval_s: float = 0.001):
        self._session = session
        self._poll_interval_s = float(poll_interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[PollSample] = []

    def __enter__(self) -> 'BackgroundPoller':
        self.start()
        return self

    def __exit__(self, *exc):
        # Best-effort: never let an exception in the foreground hang the thread.
        self.stop(timeout_s=2.0)

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(
            target=self._run, name='nvml-monitor-poll', daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        self._thread = None

    def _run(self):
        s = self._session
        stop = self._stop
        interval = self._poll_interval_s
        out = self._samples
        next_ts = time.monotonic()
        while not stop.is_set():
            try:
                ts = time.monotonic_ns()
                mhz = s.sm_clock_mhz()
                bits = s.throttle_reasons_bitmask()
                mw = s.power_milliwatts()
            except Exception:
                # NVML can transiently fail mid-run; skip the sample rather
                # than killing the poll loop.
                continue
            out.append(PollSample(
                timestamp_ns=ts,
                sm_mhz=mhz,
                throttle_bitmask=bits,
                power_watts=mw / 1000.0,
            ))
            next_ts += interval
            sleep_for = next_ts - time.monotonic()
            if sleep_for > 0:
                if stop.wait(sleep_for):
                    break
            else:
                # Behind schedule (most likely the NVML cache update period is
                # longer than the requested interval). Resync the next-tick
                # anchor so we don't accumulate negative slack indefinitely.
                next_ts = time.monotonic()

    # ---- summary --------------------------------------------------------

    def summarize(self) -> PollingResult:
        samples = self._samples
        result = PollingResult(samples=list(samples))
        if len(samples) < 2:
            result.short_window_warning = (
                f'only {len(samples)} sample(s) collected; the polled window was'
                ' shorter than the NVML driver cache update period - throttle'
                ' shares are not reliable.'
            )
            return result
        seconds: Dict[str, float] = {name: 0.0 for name, _ in THROTTLE_REASONS}
        total = 0.0
        for cur, nxt in zip(samples, samples[1:]):
            dt = max(0.0, (nxt.timestamp_ns - cur.timestamp_ns) * 1e-9)
            total += dt
            bits = cur.throttle_bitmask
            for name, mask in THROTTLE_REASONS:
                if bits & mask:
                    seconds[name] += dt
        result.reason_seconds = seconds
        result.total_seconds = total
        return result


# ---------------------------------------------------------------------------
# Locked-clock context manager (subprocess wrapper around nvidia-smi -lgc)
# ---------------------------------------------------------------------------


class LockedClock:
    """``nvidia-smi -lgc {mhz},{mhz}`` for ``device_index``, ``-rgc`` on exit.

    Used by experiment B. The caller is responsible for coordinating across
    ranks - typically only rank 0 enters the context, followed by a
    ``dist.barrier()`` so the other ranks wait for the system-wide change.

    When the lock command fails (no sudo / shared GPU), ``self.applied``
    stays ``False`` and the context yields normally so the caller can
    continue with a free-running baseline and report the skip cleanly.
    """

    def __init__(self, device_index: int, target_mhz: int):
        self.device_index = int(device_index)
        self.target_mhz = int(target_mhz)
        self.applied = False
        self.error_message: Optional[str] = None

    def __enter__(self) -> 'LockedClock':
        if self.target_mhz <= 0:
            return self
        cmd = [
            'nvidia-smi',
            '-i', str(self.device_index),
            '-lgc', f'{self.target_mhz},{self.target_mhz}',
        ]
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired) as ex:
            self.error_message = f'nvidia-smi -lgc failed to launch: {ex}'
            return self
        if cp.returncode != 0:
            self.error_message = (
                f'nvidia-smi -lgc returned {cp.returncode}: '
                f'{(cp.stderr or cp.stdout).strip() or "<no stderr>"} '
                '(needs sudo / not allowed on shared GPUs - skipping locked run)'
            )
            return self
        self.applied = True
        return self

    def __exit__(self, *exc):
        if not self.applied:
            return
        cmd = ['nvidia-smi', '-i', str(self.device_index), '-rgc']
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception:
            pass
        self.applied = False


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------


@dataclass
class IterationRow:
    iteration: int
    wall_time_us: float
    avg_sm_mhz: float            # NVML reading right after the launch
    violation_delta_ns: int      # cumulative violation diff over the iter
    power_watts: float           # NVML power right after the launch
    kernel_internal_mhz: float = float('nan')  # from clock64()/globaltimer


def _kernel_internal_median_mhz(prof_buf, num_sms: int) -> float:
    """Median per-SM MHz parsed from the ``prof_buf`` layout used in
    ``test_mega_moe.py`` (4 uint64 per SM for kernel span: clk_s, gt_s,
    clk_e, gt_e)."""
    if prof_buf is None:
        return float('nan')
    try:
        import numpy as np
        import statistics
    except Exception:
        return float('nan')
    raw = prof_buf.cpu().numpy().view(np.uint64)
    if raw.size < num_sms * 4:
        return float('nan')
    kern = raw[:num_sms * 4].reshape(num_sms, 4)
    mhz_vals: List[float] = []
    for sm in range(num_sms):
        cs, gs, ce, ge = int(kern[sm, 0]), int(kern[sm, 1]), int(kern[sm, 2]), int(kern[sm, 3])
        if ce > cs and ge > gs:
            mhz_vals.append((ce - cs) * 1000.0 / (ge - gs))
    if not mhz_vals:
        return float('nan')
    return statistics.median(mhz_vals)


def run_violation_scatter(
    *,
    session: NvmlSession,
    run_fused: Callable[..., object],
    n_iter: int,
    sync: Callable[[], None],
    prof_buf=None,
    num_sms: int = 0,
) -> List[IterationRow]:
    """Per-iteration violation snapshot and wall-clock timing.

    When ``prof_buf`` is supplied, each iteration also re-runs the kernel with
    the profiler attached so the kernel-internal MHz can be captured for cross
    check (CSV column ``kernel_internal_mhz``).
    """
    import torch
    rows: List[IterationRow] = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for i in range(n_iter):
        before_ns = session.violation_status_power_ns()
        start_evt.record()
        if prof_buf is not None:
            run_fused(profiler_buf=prof_buf)
        else:
            run_fused()
        end_evt.record()
        sync()
        wall_us = start_evt.elapsed_time(end_evt) * 1000.0  # ms -> us
        after_ns = session.violation_status_power_ns()
        delta = max(0, after_ns - before_ns)
        nvml_mhz = float(session.sm_clock_mhz())
        nvml_w = session.power_milliwatts() / 1000.0
        kern_mhz = _kernel_internal_median_mhz(prof_buf, num_sms) if prof_buf is not None else float('nan')
        rows.append(IterationRow(
            iteration=i,
            wall_time_us=wall_us,
            avg_sm_mhz=nvml_mhz,
            violation_delta_ns=delta,
            power_watts=nvml_w,
            kernel_internal_mhz=kern_mhz,
        ))
        if prof_buf is not None:
            prof_buf.zero_()
    return rows


def run_polled_window(
    *,
    session: NvmlSession,
    run_fused: Callable[..., object],
    n_iter: int,
    sync: Callable[[], None],
    poll_interval_s: float = 0.001,
) -> PollingResult:
    """Run ``n_iter`` kernels back-to-back with a background poller attached."""
    poller = BackgroundPoller(session, poll_interval_s=poll_interval_s)
    with poller:
        for _ in range(n_iter):
            run_fused()
        sync()
    return poller.summarize()


def run_locked_baseline(
    *,
    run_fused: Callable[..., object],
    n_iter: int,
    sync: Callable[[], None],
) -> Tuple[float, List[float]]:
    """Plain timing loop (no NVML reads). Returns (mean_us, per_iter_us)."""
    import torch
    per_iter: List[float] = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(n_iter):
        start_evt.record()
        run_fused()
        end_evt.record()
        sync()
        per_iter.append(start_evt.elapsed_time(end_evt) * 1000.0)
    mean_us = sum(per_iter) / len(per_iter) if per_iter else 0.0
    return mean_us, per_iter


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _open_no_clobber(path: str):
    """Open ``path`` for writing, raising if it already exists.

    The plan's AC-6 reverse test requires that we do not silently overwrite
    existing output. The caller can recover by passing a different
    ``--nvml-out-dir``.
    """
    if os.path.exists(path):
        raise FileExistsError(
            f'NVML output file already exists: {path} '
            '(use a different --nvml-out-dir or delete the old one)'
        )
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    return open(path, 'w', newline='')


def write_iteration_csv(path: str, rows: Sequence[IterationRow]):
    with _open_no_clobber(path) as f:
        w = csv.writer(f)
        w.writerow([
            'iteration', 'wall_time_us', 'avg_sm_mhz', 'violation_delta_ns',
            'power_watts', 'kernel_internal_mhz',
        ])
        for r in rows:
            w.writerow([
                r.iteration, f'{r.wall_time_us:.3f}', f'{r.avg_sm_mhz:.1f}',
                r.violation_delta_ns, f'{r.power_watts:.3f}',
                f'{r.kernel_internal_mhz:.1f}' if r.kernel_internal_mhz == r.kernel_internal_mhz else '',
            ])


def write_polling_csv(path: str, polling: PollingResult, summary_path: str):
    with _open_no_clobber(path) as f:
        w = csv.writer(f)
        w.writerow(['timestamp_ns', 'sm_mhz', 'throttle_bitmask', 'power_watts'])
        for s in polling.samples:
            w.writerow([s.timestamp_ns, s.sm_mhz, s.throttle_bitmask, f'{s.power_watts:.3f}'])
    with _open_no_clobber(summary_path) as f:
        w = csv.writer(f)
        w.writerow(['reason', 'seconds', 'share'])
        total = polling.total_seconds
        for name, _ in THROTTLE_REASONS:
            secs = polling.reason_seconds.get(name, 0.0)
            share = (secs / total) if total > 0 else 0.0
            w.writerow([name, f'{secs:.6f}', f'{share:.4f}'])
        w.writerow(['__total__', f'{total:.6f}', '1.0'])
        if polling.short_window_warning:
            w.writerow(['__warning__', polling.short_window_warning, ''])


def write_locked_compare_csv(
    path: str,
    *,
    free_mean_us: float,
    locked_mean_us: float,
    free_mhz_median: float,
    locked_mhz_median: float,
    locked_applied: bool,
    locked_error: Optional[str],
):
    with _open_no_clobber(path) as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['free_mean_us', f'{free_mean_us:.3f}'])
        w.writerow(['locked_mean_us', f'{locked_mean_us:.3f}'])
        w.writerow(['free_mhz_median', f'{free_mhz_median:.1f}'])
        w.writerow(['locked_mhz_median', f'{locked_mhz_median:.1f}'])
        w.writerow(['locked_applied', 'true' if locked_applied else 'false'])
        if locked_error:
            w.writerow(['locked_error', locked_error])
        if locked_applied and locked_mean_us > 0 and locked_mhz_median > 0:
            perf_loss = (free_mean_us - locked_mean_us) / locked_mean_us
            freq_drop = (locked_mhz_median - free_mhz_median) / locked_mhz_median
            sensitivity = (perf_loss / freq_drop) if abs(freq_drop) > 1e-9 else float('nan')
            w.writerow(['perf_loss', f'{perf_loss:.4f}'])
            w.writerow(['freq_drop', f'{freq_drop:.4f}'])
            w.writerow(['freq_sensitivity', f'{sensitivity:.4f}'])


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class NvmlMonitorConfig:
    out_dir: str
    n_iter_violation: int = 20
    n_iter_polled: int = 40
    n_iter_locked: int = 20
    poll_interval_s: float = 0.001
    locked_mhz: int = 0        # 0 disables experiment B
    cross_check_with_profiler: bool = True


def _median(values: Sequence[float]) -> float:
    import statistics
    return statistics.median(values) if values else float('nan')


def run_nvml_monitor(
    *,
    config: NvmlMonitorConfig,
    rank_idx: int,
    local_rank: int,
    is_distributed_leader: bool,
    distributed_barrier: Callable[[], None],
    run_fused: Callable[..., object],
    sync: Callable[[], None],
    prof_buf=None,
    num_sms: int = 0,
    log: Callable[[str], None] = print,
) -> Dict[str, str]:
    """Drive the four experiments and write per-rank CSVs.

    Returns a dict of ``{experiment_label: csv_path}`` for the caller's
    bookkeeping.
    """
    paths: Dict[str, str] = {}

    with NvmlSession(device_index=local_rank) as session:
        log(f'[nvml rank {rank_idx}] device_index={session.device_index} '
            f'initial_sm_mhz={session.sm_clock_mhz()} power_w={session.power_milliwatts()/1000:.1f}')

        # --- Violation scatter (experiment A) ---
        cross_buf = prof_buf if (config.cross_check_with_profiler and prof_buf is not None) else None
        rows = run_violation_scatter(
            session=session, run_fused=run_fused, n_iter=config.n_iter_violation,
            sync=sync, prof_buf=cross_buf, num_sms=num_sms,
        )
        iter_path = os.path.join(config.out_dir, f'nvml_rank{rank_idx}_violation.csv')
        write_iteration_csv(iter_path, rows)
        paths['violation_scatter'] = iter_path

        # Cross-check report (experiment A + AC-8): compare NVML SM clock
        # median to kernel-internal MHz median over the same iteration window.
        nvml_mhzs = [r.avg_sm_mhz for r in rows]
        kern_mhzs = [r.kernel_internal_mhz for r in rows if r.kernel_internal_mhz == r.kernel_internal_mhz]
        nvml_med = _median(nvml_mhzs)
        kern_med = _median(kern_mhzs) if kern_mhzs else float('nan')
        if kern_mhzs and kern_med == kern_med and nvml_med == nvml_med and kern_med > 0:
            delta_pct = (nvml_med - kern_med) / kern_med * 100.0
            log(f'[nvml rank {rank_idx}] cross-check: NVML SM median={nvml_med:.0f} MHz, '
                f'kernel-internal median={kern_med:.0f} MHz, delta={delta_pct:+.2f}%')
        else:
            log(f'[nvml rank {rank_idx}] cross-check: NVML median={nvml_med:.0f} MHz '
                '(no kernel-internal MHz available - profiler buffer disabled or empty)')

        # --- Polled window (experiments C + D) ---
        polling = run_polled_window(
            session=session, run_fused=run_fused,
            n_iter=config.n_iter_polled, sync=sync,
            poll_interval_s=config.poll_interval_s,
        )
        poll_path = os.path.join(config.out_dir, f'nvml_rank{rank_idx}_poll_timeseries.csv')
        poll_summary_path = os.path.join(config.out_dir, f'nvml_rank{rank_idx}_throttle_summary.csv')
        write_polling_csv(poll_path, polling, poll_summary_path)
        paths['poll_timeseries'] = poll_path
        paths['throttle_summary'] = poll_summary_path

        if polling.short_window_warning:
            log(f'[nvml rank {rank_idx}] {polling.short_window_warning}')
        else:
            top_reasons = sorted(
                ((name, polling.reason_seconds.get(name, 0.0)) for name, _ in THROTTLE_REASONS),
                key=lambda kv: -kv[1],
            )[:3]
            pretty = ', '.join(
                f'{n}: {(s/polling.total_seconds*100 if polling.total_seconds>0 else 0):.1f}%'
                for n, s in top_reasons
            )
            log(f'[nvml rank {rank_idx}] throttle share over {polling.total_seconds*1000:.0f} ms '
                f'({len(polling.samples)} samples): {pretty}')

        # --- Locked-clock comparison (experiment B) ---
        if config.locked_mhz > 0:
            free_mean_us, _ = run_locked_baseline(
                run_fused=run_fused, n_iter=config.n_iter_locked, sync=sync,
            )
            free_mhz_med = float(session.sm_clock_mhz())
            distributed_barrier()
            lc = LockedClock(device_index=local_rank, target_mhz=config.locked_mhz)
            locked_mean_us = 0.0
            locked_mhz_med = float('nan')
            if is_distributed_leader:
                lc.__enter__()
            distributed_barrier()
            try:
                if lc.error_message and is_distributed_leader:
                    log(f'[nvml rank {rank_idx}] locked-clock baseline skipped: {lc.error_message}')
                # Even when the lock command failed, run the timing loop so we
                # surface a free-vs-free row rather than no row at all - the
                # CSV records locked_applied=false for the analyst.
                locked_mean_us, _ = run_locked_baseline(
                    run_fused=run_fused, n_iter=config.n_iter_locked, sync=sync,
                )
                locked_mhz_med = float(session.sm_clock_mhz())
            finally:
                distributed_barrier()
                if is_distributed_leader:
                    lc.__exit__(None, None, None)
                distributed_barrier()
            locked_path = os.path.join(config.out_dir, f'nvml_rank{rank_idx}_locked_compare.csv')
            write_locked_compare_csv(
                locked_path,
                free_mean_us=free_mean_us, locked_mean_us=locked_mean_us,
                free_mhz_median=free_mhz_med, locked_mhz_median=locked_mhz_med,
                locked_applied=lc.applied or False,
                locked_error=lc.error_message,
            )
            paths['locked_compare'] = locked_path

    return paths
