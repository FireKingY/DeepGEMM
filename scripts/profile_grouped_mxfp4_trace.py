import argparse
import ctypes
import ctypes.util
from collections import defaultdict
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_ROOT = os.path.join(REPO_ROOT, "tests")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TESTS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_ROOT)

import deep_gemm

from generators import (  # noqa: E402
    KernelType,
    MajorTypeAB,
    QuantConfig,
    align,
    cast_fp8_fp4_with_major,
    get_ue8m0_usage,
    grouped_cast_fp8_fp4_with_major,
)
from deep_gemm.utils import get_mk_alignment_for_contiguous_layout  # noqa: E402

NUM_WARPS_PER_BLOCK = 8
EXPORT_TRACKS_PER_CTA = 64


def stage_tags(prefix: str) -> list[str]:
    return [f"{prefix}_STAGE{i}" for i in range(32)]


TAGS = [
    "INIT",
    "LOAD_WAIT_EMPTY",
    "LOAD_TMA_ISSUE",
    *stage_tags("TMA_TO_SF_READY"),
    "SF_WAIT_FULL",
    "SF_TRANSPOSE",
    "MMA_WAIT_TMEM_EMPTY",
    "MMA_WAIT_READY",
    "UTCCP_SCALE",
    "UMMA",
    *stage_tags("UMMA_TO_EMPTY_READY"),
    "EPILOGUE_WAIT_FULL",
    "EPILOGUE_STORE",
]
ROLE_NAMES = {
    0: "load",
    1: "mma",
    2: "sf",
    3: "idle",
    4: "epilogue0",
    5: "epilogue1",
    6: "epilogue2",
    7: "epilogue3",
}


def remap_trace_track(slot_id: int, event_name: str) -> tuple[int, str]:
    slot_base = slot_id - (slot_id % NUM_WARPS_PER_BLOCK)
    cta_idx = slot_base // NUM_WARPS_PER_BLOCK
    export_base = cta_idx * EXPORT_TRACKS_PER_CTA
    warp_role = slot_id % NUM_WARPS_PER_BLOCK
    if event_name.startswith("TMA_TO_SF_READY_STAGE"):
        stage = int(event_name.rsplit("STAGE", 1)[1])
        return export_base + 3 + stage, f"tma_to_sf_ready_stage{stage}"
    if event_name.startswith("UMMA_TO_EMPTY_READY_STAGE"):
        return export_base + 35, "umma_to_empty_ready"
    if warp_role == 0:
        return export_base + 0, "load"
    if warp_role == 1:
        return export_base + 1, "mma"
    if warp_role == 2:
        return export_base + 2, "sf"
    if warp_role == 3:
        return export_base + 36, "idle"
    if warp_role in (4, 5, 6, 7):
        return export_base + 37 + (warp_role - 4), ROLE_NAMES[warp_role]
    return export_base + 63, ROLE_NAMES.get(warp_role, f"warp{warp_role}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile grouped MXFP4 GEMM and export chrome trace.")
    parser.add_argument("--num-groups", type=int, default=32)
    parser.add_argument("--tokens-per-expert", type=int, default=1024)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--pre-profile-sleep-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench-iters", type=int, default=0)
    parser.add_argument("--bench-samples", type=int, default=3)
    parser.add_argument("--trace-num-entries", type=int, default=4096)
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--trace-sm-id", type=int, default=-1)
    parser.add_argument("--mainloop-num-stages", type=int, default=3)
    parser.add_argument("--epilogue-num-stages", type=int, default=2)
    parser.add_argument("--skip-profile-window", action="store_true")
    return parser.parse_args()


def load_cudart() -> ctypes.CDLL:
    candidates = []
    libname = ctypes.util.find_library("cudart")
    if libname:
        candidates.append(libname)
    candidates.extend((
        "libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda-13.1/lib64/libcudart.so",
        "/usr/local/cuda-12.8/lib64/libcudart.so",
    ))
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise RuntimeError("Failed to load libcudart.so")


def check_cuda(err: int, op: str) -> None:
    if err != 0:
        raise RuntimeError(f"{op} failed with CUDA error code {err}")


def generate_fixed_case(
    num_groups: int,
    tokens_per_expert: int,
    n: int,
    k: int,
):
    quant_config = QuantConfig((32, 32, True, True))
    major_a = MajorTypeAB.KMajor
    major_b = MajorTypeAB.KMajor
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    aligned_m = align(tokens_per_expert, get_mk_alignment_for_contiguous_layout())
    total_m = aligned_m * num_groups

    a = torch.randn((total_m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    grouped_layout = torch.empty(total_m, device="cuda", dtype=torch.int32)
    d = torch.empty((total_m, n), device="cuda", dtype=torch.bfloat16)

    start = 0
    for group_idx in range(num_groups):
        actual_end = start + tokens_per_expert
        aligned_end = start + aligned_m
        grouped_layout[start:actual_end] = group_idx
        grouped_layout[actual_end:aligned_end] = -1
        a[actual_end:aligned_end] = 0
        start = aligned_end

    a_q = cast_fp8_fp4_with_major(
        a, major_a, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0
    )
    b_q = grouped_cast_fp8_fp4_with_major(
        b,
        major_b,
        quant_config.gran_k_b,
        quant_config.is_fp4_b,
        use_ue8m0,
        use_block_cast_for_fp8=True,
    )
    return total_m, aligned_m, a_q, b_q, grouped_layout, d, quant_config, (not use_ue8m0)


def normalize_trace_path(trace_out: Path) -> Path:
    trace_out = trace_out.resolve()
    if trace_out.suffix == ".gz":
        return trace_out.with_suffix("")
    if trace_out.suffix == ".json":
        return trace_out
    return trace_out.with_suffix(".json")


def parse_stage_suffix(event_name: str, prefix: str) -> int | None:
    if not event_name.startswith(prefix):
        return None
    return int(event_name.rsplit("STAGE", 1)[1])


def ordered_events(events: list[dict], priority_map: dict[str, int] | None = None) -> list[dict]:
    priority_map = priority_map or {}
    return sorted(events, key=lambda item: (item["start_ns"], priority_map.get(item["name"], 1000), item["name"]))


def annotate_runtime_stage_tags(
    raw_events: list[dict],
    mainloop_num_stages: int,
    epilogue_num_stages: int,
    sf_refresh_period_a: int = 1,
    sf_refresh_period_b: int = 1,
) -> list[dict]:
    events_by_cta: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for event in raw_events:
        slot_base = event["slot_id"] - (event["slot_id"] % NUM_WARPS_PER_BLOCK)
        events_by_cta[(event["sm_id"], slot_base)].append(event)

    annotated_events: list[dict] = []
    sf_refresh_period_a = max(sf_refresh_period_a, 1)
    sf_refresh_period_b = max(sf_refresh_period_b, 1)

    for (_, slot_base), cta_events in events_by_cta.items():
        events_by_slot: dict[int, list[dict]] = defaultdict(list)
        for event in cta_events:
            events_by_slot[event["slot_id"]].append(event)

        load_slot = slot_base + 0
        load_wait_counter = 0
        for event in sorted(events_by_slot.get(load_slot, []), key=lambda item: (item["start_ns"], item["name"])):
            annotated_event = dict(event)
            if event["name"] == "LOAD_WAIT_EMPTY":
                stage = load_wait_counter % mainloop_num_stages
                annotated_event["name"] = f"LOAD_WAIT_EMPTY_STAGE{stage}"
                load_wait_counter += 1
            annotated_events.append(annotated_event)

        mma_slot = slot_base + 1
        mma_wait_ready_counter = 0
        mma_wait_tmem_counter = 0
        for event in sorted(events_by_slot.get(mma_slot, []), key=lambda item: (item["start_ns"], item["name"])):
            annotated_event = dict(event)
            if event["name"] == "MMA_WAIT_READY":
                stage = mma_wait_ready_counter % mainloop_num_stages
                annotated_event["name"] = f"MMA_WAIT_READY_STAGE{stage}"
                mma_wait_ready_counter += 1
            elif event["name"] == "MMA_WAIT_TMEM_EMPTY":
                stage = mma_wait_tmem_counter % epilogue_num_stages
                annotated_event["name"] = f"MMA_WAIT_TMEM_EMPTY_STAGE{stage}"
                mma_wait_tmem_counter += 1
            annotated_events.append(annotated_event)

        sf_slot = slot_base + 2
        sf_stage_sequence = [
            k_block_idx % mainloop_num_stages
            for k_block_idx in range(load_wait_counter)
            if (k_block_idx % sf_refresh_period_a) == 0 or (k_block_idx % sf_refresh_period_b) == 0
        ]
        sf_wait_counter = 0
        for event in sorted(events_by_slot.get(sf_slot, []), key=lambda item: (item["start_ns"], item["name"])):
            annotated_event = dict(event)
            if event["name"] == "SF_WAIT_FULL":
                if sf_wait_counter >= len(sf_stage_sequence):
                    raise RuntimeError("SF_WAIT_FULL count exceeded inferred refresh schedule")
                stage = sf_stage_sequence[sf_wait_counter]
                annotated_event["name"] = f"SF_WAIT_FULL_STAGE{stage}"
                sf_wait_counter += 1
            annotated_events.append(annotated_event)

        for slot_id, slot_events in events_by_slot.items():
            if slot_id in (load_slot, mma_slot, sf_slot):
                continue
            annotated_events.extend(sorted((dict(event) for event in slot_events), key=lambda item: (item["start_ns"], item["name"])))

    return annotated_events


def extract_wait_ends_by_stage(events: list[dict], prefix: str) -> dict[int, list[int]]:
    ends_by_stage: dict[int, list[int]] = defaultdict(list)
    for event in ordered_events(events):
        stage = parse_stage_suffix(event["name"], prefix)
        if stage is not None:
            ends_by_stage[stage].append(event["start_ns"] + event["duration_ns"])
    return ends_by_stage


def extract_load_tma_issue_starts_by_stage(events: list[dict]) -> dict[int, list[int]]:
    starts_by_stage: dict[int, list[int]] = defaultdict(list)
    current_stage: int | None = None
    priorities = {f"LOAD_WAIT_EMPTY_STAGE{i}": 0 for i in range(32)}
    priorities["LOAD_TMA_ISSUE"] = 1
    for event in ordered_events(events, priorities):
        stage = parse_stage_suffix(event["name"], "LOAD_WAIT_EMPTY_STAGE")
        if stage is not None:
            current_stage = stage
        elif event["name"] == "LOAD_TMA_ISSUE" and current_stage is not None:
            starts_by_stage[current_stage].append(event["start_ns"])
    return starts_by_stage


def extract_umma_starts_by_stage(events: list[dict], mainloop_num_stages: int) -> dict[int, list[int]]:
    starts_by_stage: dict[int, list[int]] = defaultdict(list)
    current_stage: int | None = None
    priorities = {f"MMA_WAIT_READY_STAGE{i}": 0 for i in range(32)}
    priorities["UMMA"] = 1
    saw_explicit_stage = False
    fallback_umma_counter = 0
    for event in ordered_events(events, priorities):
        stage = parse_stage_suffix(event["name"], "MMA_WAIT_READY_STAGE")
        if stage is not None:
            current_stage = stage
            saw_explicit_stage = True
        elif event["name"] == "UMMA" and current_stage is not None:
            starts_by_stage[current_stage].append(event["start_ns"])
        elif event["name"] == "UMMA" and not saw_explicit_stage:
            stage = fallback_umma_counter % max(mainloop_num_stages, 1)
            starts_by_stage[stage].append(event["start_ns"])
            fallback_umma_counter += 1
    return starts_by_stage


def synthesize_crosswarp_events(raw_events: list[dict], mainloop_num_stages: int) -> list[dict]:
    events_by_cta: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for event in raw_events:
        slot_id = event["slot_id"]
        slot_base = slot_id - (slot_id % NUM_WARPS_PER_BLOCK)
        events_by_cta[(event["sm_id"], slot_base)].append(event)

    synthetic_events: list[dict] = []
    for (sm_id, slot_base), cta_events in events_by_cta.items():
        events_by_slot: dict[int, list[dict]] = defaultdict(list)
        for event in cta_events:
            events_by_slot[event["slot_id"]].append(event)

        load_events = events_by_slot.get(slot_base + 0, [])
        mma_events = events_by_slot.get(slot_base + 1, [])
        sf_events = events_by_slot.get(slot_base + 2, [])

        load_tma_issue_starts = extract_load_tma_issue_starts_by_stage(load_events)
        sf_wait_ends = extract_wait_ends_by_stage(sf_events, "SF_WAIT_FULL_STAGE")
        for stage, start_list in load_tma_issue_starts.items():
            end_list = sf_wait_ends.get(stage, [])
            for start_ns, end_ns in zip(start_list, end_list):
                if end_ns >= start_ns:
                    synthetic_events.append(
                        {
                            "slot_id": slot_base,
                            "sm_id": sm_id,
                            "name": f"TMA_TO_SF_READY_STAGE{stage}",
                            "start_ns": start_ns,
                            "duration_ns": end_ns - start_ns,
                        }
                    )

        empty_ready_ends = extract_wait_ends_by_stage(load_events, "LOAD_WAIT_EMPTY_STAGE")
        umma_starts = extract_umma_starts_by_stage(mma_events, mainloop_num_stages)

        umma_to_empty_candidates: list[dict] = []
        for stage in range(32):
            start_list = umma_starts.get(stage, [])
            end_list = empty_ready_ends.get(stage, [])
            pair_count = min(len(start_list), max(len(end_list) - 1, 0))
            for idx in range(pair_count):
                umma_to_empty_candidates.append(
                    {
                        "stage": stage,
                        "issue_start_ns": start_list[idx],
                        "end_ns": end_list[idx + 1],
                    }
                )

        prev_end_ns = 0
        for candidate in sorted(umma_to_empty_candidates, key=lambda item: (item["issue_start_ns"], item["stage"])):
            start_ns = max(prev_end_ns, candidate["issue_start_ns"])
            end_ns = candidate["end_ns"]
            if end_ns >= start_ns:
                synthetic_events.append(
                    {
                        "slot_id": slot_base,
                        "sm_id": sm_id,
                        "name": f"UMMA_TO_EMPTY_READY_STAGE{candidate['stage']}",
                        "start_ns": start_ns,
                        "duration_ns": end_ns - start_ns,
                    }
                )
                prev_end_ns = end_ns

    return synthetic_events


def export_trace(
    profiler: torch.Tensor,
    trace_out: Path,
    trace_sm_id: int,
    mainloop_num_stages: int,
    epilogue_num_stages: int,
) -> tuple[list[dict], Path]:
    rows = profiler.cpu().tolist()
    raw_events: list[dict] = []
    metadata: list[dict] = []
    named_threads: set[tuple[int, int]] = set()

    for slot_id, row in enumerate(rows):
        event_count = row[0]
        for event_idx in range(event_count):
            sm_id, tag, start_ns, duration_ns = row[1 + event_idx * 4: 1 + (event_idx + 1) * 4]
            if 0 <= tag < len(TAGS):
                event_name = TAGS[tag]
            else:
                event_name = f"TAG_{tag}"
            raw_events.append(
                {
                    "slot_id": slot_id,
                    "sm_id": sm_id,
                    "name": event_name,
                    "start_ns": start_ns,
                    "duration_ns": duration_ns,
                }
            )

    raw_events = annotate_runtime_stage_tags(
        raw_events,
        mainloop_num_stages=mainloop_num_stages,
        epilogue_num_stages=epilogue_num_stages,
        sf_refresh_period_a=1,
        sf_refresh_period_b=1,
    )
    raw_events.extend(synthesize_crosswarp_events(raw_events, mainloop_num_stages))
    raw_events = [event for event in raw_events if event["name"] != "INIT"]
    if trace_sm_id >= 0:
        raw_events = [event for event in raw_events if event["sm_id"] == trace_sm_id]

    base_start_ns = min((event["start_ns"] for event in raw_events), default=0)
    events: list[dict] = []
    for raw_event in sorted(raw_events, key=lambda item: (item["sm_id"], item["slot_id"], item["start_ns"], item["name"])):
        slot_id = raw_event["slot_id"]
        sm_id = raw_event["sm_id"]
        event_name = raw_event["name"]
        export_tid, role_name = remap_trace_track(slot_id, event_name)
        if (sm_id, export_tid) not in named_threads:
            metadata.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": sm_id,
                    "tid": export_tid,
                    "args": {"name": f"slot{export_tid}:{role_name}"},
                }
            )
            named_threads.add((sm_id, export_tid))
        events.append(
            {
                "name": event_name,
                "ph": "X",
                "ts": (raw_event["start_ns"] - base_start_ns) / 1e3,
                "dur": raw_event["duration_ns"] / 1e3,
                "pid": sm_id,
                "tid": export_tid,
            }
        )

    trace_out = normalize_trace_path(trace_out)
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_out, "w", encoding="utf-8") as f:
        json.dump({"traceEvents": metadata + events}, f)
    return events, trace_out


def summarize_profiler_rows(profiler: torch.Tensor) -> tuple[int, int, dict[str, int], dict[str, int]]:
    rows = profiler.cpu().tolist()
    max_entries = (profiler.size(1) - 1) // 4
    saturated_slots = 0
    max_counts_by_role: dict[str, int] = {}
    saturated_by_role: dict[str, int] = {}

    for slot_id, row in enumerate(rows):
        role_name = ROLE_NAMES.get(slot_id % NUM_WARPS_PER_BLOCK, f"warp{slot_id % NUM_WARPS_PER_BLOCK}")
        event_count = row[0]
        max_counts_by_role[role_name] = max(max_counts_by_role.get(role_name, 0), event_count)
        if event_count >= max_entries:
            saturated_slots += 1
            saturated_by_role[role_name] = saturated_by_role.get(role_name, 0) + 1

    return saturated_slots, max_entries, max_counts_by_role, saturated_by_role


def total_duration_us_by_prefix(events: list[dict], prefix: str) -> float:
    return sum(event["dur"] for event in events if event["name"].startswith(prefix))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    total_m, aligned_m, a, b, grouped_layout, d, quant_config, disable_ue8m0_cast = generate_fixed_case(
        num_groups=args.num_groups,
        tokens_per_expert=args.tokens_per_expert,
        n=args.n,
        k=args.k,
    )
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    def launch(profiler: torch.Tensor | None = None) -> None:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a,
            b,
            d,
            grouped_layout,
            disable_ue8m0_cast=disable_ue8m0_cast,
            use_psum_layout=False,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            use_mxfp4=True,
            profiler=profiler,
        )

    print(
        f"profile target: combo=fp4xfp4_mxfp4 G={args.num_groups} "
        f"actual_m_per_group={args.tokens_per_expert} aligned_m_per_group={aligned_m} "
        f"M={total_m} N={args.n} K={args.k} out=bf16"
    )
    print(f"device: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")

    launch()
    torch.cuda.synchronize()

    for _ in range(args.warmup_iters):
        launch()
    torch.cuda.synchronize()

    if args.bench_iters > 0:
        samples = []
        for sample_idx in range(args.bench_samples):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(args.bench_iters):
                launch()
            end_event.record()
            end_event.synchronize()
            elapsed_us = start_event.elapsed_time(end_event) * 1000.0 / args.bench_iters
            samples.append(elapsed_us)
            print(f"bench sample {sample_idx + 1}: {elapsed_us:.3f} us")
        mean_us = sum(samples) / len(samples)
        stdev_us = (sum((sample - mean_us) ** 2 for sample in samples) / len(samples)) ** 0.5
        flops_tflops = 2 * total_m * args.n * args.k / (mean_us * 1e-6) / 1e12
        print(f"bench mean: {mean_us:.3f} us")
        print(f"bench stdev: {stdev_us:.3f} us")
        print(f"bench achieved: {flops_tflops:.3f} TFLOPS")

    if args.trace_out is not None:
        num_slots = deep_gemm.get_num_sms() * NUM_WARPS_PER_BLOCK
        profiler = torch.zeros(
            (num_slots, 1 + args.trace_num_entries * 4),
            dtype=torch.int64,
            device="cuda",
        )
        torch.cuda.synchronize()
        launch(profiler)
        torch.cuda.synchronize()

        trace_events, trace_path = export_trace(
            profiler,
            args.trace_out,
            args.trace_sm_id,
            args.mainloop_num_stages,
            args.epilogue_num_stages,
        )
        saturated_slots, max_entries, max_counts_by_role, saturated_by_role = summarize_profiler_rows(profiler)
        print(f"trace_out: {trace_path}")
        print(f"trace events: {len(trace_events)}")
        print(f"trace max_entries_per_slot: {max_entries}")
        print(f"trace saturated slots: {saturated_slots}/{num_slots}")
        print(f"trace max event count by role: {max_counts_by_role}")
        if saturated_by_role:
            print(f"trace saturated slots by role: {saturated_by_role}")
        print(f"tma_to_sf_ready_total_us={total_duration_us_by_prefix(trace_events, 'TMA_TO_SF_READY_STAGE'):.3f}")
        print(f"umma_to_empty_ready_total_us={total_duration_us_by_prefix(trace_events, 'UMMA_TO_EMPTY_READY_STAGE'):.3f}")

    if args.skip_profile_window:
        return

    if args.pre_profile_sleep_s > 0:
        time.sleep(args.pre_profile_sleep_s)

    cudart = load_cudart()
    check_cuda(cudart.cudaProfilerStart(), "cudaProfilerStart")
    for _ in range(args.profile_iters):
        launch()
    torch.cuda.synchronize()
    check_cuda(cudart.cudaProfilerStop(), "cudaProfilerStop")
    print("profile window completed")


if __name__ == "__main__":
    main()
