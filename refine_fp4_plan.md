# Add a Dedicated SM100 MXFP4 Grouped GEMM Path with JIT Dispatch

## Goal Description
Implement a new SM100 grouped GEMM path for MXFP4 block-scaled MMA in DeepGEMM without modifying the existing grouped FP8/FP4 device kernel. The new path should live behind distinct JIT/runtime artifacts, activate only when the caller explicitly requests MXFP4, and preserve current behavior for all existing grouped FP8xFP8 / FP8xFP4 flows when that opt-in is not provided.

Phase 1 is intentionally narrow: target `GemmType::MGroupedContiguous` first, keep BF16 output and packed UE8M0 scales, require both operands to be FP4, require an explicit MXFP4 opt-in parameter at the API boundary, and leave masked / psum / broader layout expansion for follow-up work unless the user explicitly broadens scope.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: A separate MXFP4 grouped kernel stack exists, with isolated JIT artifacts and no behavior change to the existing grouped FP8/FP4 path.
  - Positive Tests (expected to PASS):
    - An eligible SM100 grouped FP4xFP4 contiguous case JIT-compiles and launches through a new runtime/build target such as `sm100_m_grouped_mxfp4_gemm_contiguous_1d1d`.
    - Existing grouped FP8xFP8 / FP8xFP4 cases still compile and route through the current runtime names, including `sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d` and `sm100_m_grouped_fp8_fp4_gemm_masked_1d1d`.
  - Negative Tests (expected to FAIL):
    - An eligible MXFP4 case must not silently reuse the legacy grouped runtime/build target.
    - Existing FP8xFP8 / FP8xFP4 grouped tests must not start dispatching to the new MXFP4 runtime.
  - AC-1.1: The original SM100 grouped device kernel remains untouched.
    - Positive: The new MXFP4 path is implemented in sibling files and host dispatch layers; the existing device kernel remains the reference implementation for the legacy path.
    - Negative: No implementation step depends on editing `deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh` to relax its current `BLOCK_K == 128` assumption.
- AC-2: Dispatch and legality modeling explicitly distinguish MXFP4 from MXFP8FP4.
  - Positive Tests (expected to PASS):
    - `deep_gemm/include/deep_gemm/common/types.hpp`, `csrc/jit_kernels/heuristics/common.hpp`, `csrc/jit_kernels/heuristics/sm100.hpp`, and `csrc/apis/gemm.hpp` can identify MXFP4-eligible grouped contiguous inputs and select an MXFP4-specific config with `block_k = 256`.
    - Supported phase-1 cases (SM100, grouped contiguous, both operands FP4, supported scale recipe, supported layout, supported output type, and explicit MXFP4 opt-in enabled) deterministically choose the new MXFP4 path.
  - Negative Tests (expected to FAIL):
    - SM90, grouped masked, psum-layout, unsupported layout variants, unsupported granularity combinations, or calls without the explicit MXFP4 opt-in do not dispatch into MXFP4.
    - K shapes outside the declared phase-1 policy (for example, non-multiples of 256 if that remains the launch constraint) do not silently enter an illegal MXFP4 config.
  - AC-2.1: Ineligible FP4 cases hard-fail instead of falling back to the legacy grouped path.
    - Positive: Tests verify deterministic hard-error behavior for FP4 inputs that are not MXFP4-eligible because of layout, grouped mode, scale recipe, or K-policy violations.
    - Negative: There is no silent fallback to the legacy grouped path and no silent misdispatch into a path with incompatible layout, scale, or tile-K assumptions.
- AC-3: The new runtime/device implementation encodes MXFP4-specific MMA, TMA, UTCCP, and epilogue assumptions instead of reusing legacy MXFP8FP4 constants.
  - Positive Tests (expected to PASS):
    - Low-level wrappers for `tcgen05.mma ... kind::mxf4.block_scale` exist for the supported CTA-group modes used by the new path.
    - The MXFP4 device kernel compiles with MXFP4-specific `BLOCK_K`, scale cadence, TMEM column sizing, and grouped-epilogue ownership logic for at least one eligible grouped contiguous FP4xFP4 workload.
    - Tensor-map creation for A/B/SF/CD succeeds for the declared phase-1 MXFP4 grouped layout constraints.
  - Negative Tests (expected to FAIL):
    - The new implementation must not issue `SM100_MMA_MXF8F6F4_*` instructions.
    - The new implementation must not reuse the legacy `BLOCK_K == 128` kernel template as if MXFP4 were a drop-in opcode replacement.
- AC-4: Validation covers correctness, dispatch regression, and rollout readiness.
  - Positive Tests (expected to PASS):
    - Grouped contiguous FP4xFP4 correctness tests pass across multiple group counts and shapes, including uneven group sizes and a user-relevant workload such as `num_groups=32, M=32768, N=2048, K=7168`, within the repo’s FP4 tolerance budget.
    - Existing grouped FP8xFP8 / FP8xFP4 contiguous, masked, and psum tests continue to pass without changed tolerances.
    - Benchmark and/or debug tracing can show which runtime/build target was selected, enabling apples-to-apples comparison between the new MXFP4 path and the legacy `mxf8f6f4` path on the same eligible workload.
  - Negative Tests (expected to FAIL):
    - Unsupported phase-1 cases (masked, psum, disallowed layout, unsupported K-tail policy) must not appear to pass by accidentally using the wrong runtime.
    - Default routing must not switch to MXFP4 for a workload that fails the agreed benchmark gate.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The implementation adds a distinct `MmaKind::MXFP4`, new low-level PTX wrappers, a dedicated SM100 MXFP4 grouped-contiguous device kernel, a dedicated JIT/runtime wrapper, an explicit dispatch predicate, FP4xFP4 grouped correctness tests, legacy regression tests, and benchmark/trace coverage sufficient to decide whether default dispatch should be enabled.

This completes the user-requested MXFP4 grouped path without over-engineering into masked / psum / k-grouped / layout-generalized support before the contiguous path is proven correct.

### Lower Bound (Minimum Acceptable Scope)
The implementation supports only SM100 `MGroupedContiguous` BF16-output grouped GEMM for both-operand FP4 inputs under an explicitly documented eligibility envelope (including supported layout(s), packed UE8M0 scales, phase-1 K constraints, and an explicit MXFP4 opt-in parameter), and all other explicit-MXFP4 FP4 cases hard-fail with a clearly documented error.

This is the minimum acceptable scope because it delivers a real MXFP4 grouped kernel and JIT dispatch path while keeping the blast radius small enough to validate deterministically.

### Allowed Choices
- Can use: new sibling kernel/runtime files, a new `MmaKind::MXFP4`, an explicit API parameter that opt-ins to MXFP4 dispatch, narrow phase-1 eligibility, hard errors for unsupported explicit-MXFP4 FP4 cases, and separate follow-up tasks for masked / psum / 2SM expansion.
- Cannot use: changing the behavior of the existing SM100 grouped device kernel, treating MXFP4 as a pure opcode swap inside the old kernel, mixing `mxf4` and `mxf4nvf4` in the same phase, silently widening phase-1 to masked / psum / mixed-FP8 cases without dedicated legality checks and tests, or requiring 2SM in phase 1.

> **Note on Deterministic Designs**: The user’s draft strongly constrains the design: this must be a new kernel path, not a modification of the original one. The plan therefore keeps that choice fixed and narrows phase-1 scope rather than leaving the core structure open-ended.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
One reasonable implementation path is:

1. Add an MXFP4 modeling hook.
   - Introduce `MmaKind::MXFP4` in `deep_gemm/include/deep_gemm/common/types.hpp:5-14`.
   - Extend `to_string(...)`, element-size handling, and heuristic selection so MXFP4 no longer falls into the same bucket as `MXFP8FP4`.

2. Define the phase-1 eligibility predicate close to API dispatch.
   - In `csrc/apis/gemm.hpp:143-205`, add an explicit API parameter and a helper that checks `arch == 10`, grouped-contiguous gemm type, both operands FP4, supported layout(s), supported `recipe_a/recipe_b`, BF16 output, `use_psum_layout == false`, the agreed K policy, and whether MXFP4 opt-in is enabled.
   - Route only matching opt-in cases to the new runtime; keep all non-opt-in calls on the legacy path, and reject unsupported opt-in cases per DEC-2.

3. Fork the SM100 grouped 1D1D implementation instead of patching the old one.
   - Mirror the current stack with a new device kernel file (for example `deep_gemm/include/deep_gemm/impls/sm100_mxfp4_gemm_1d1d.cuh`) and a matching host/JIT wrapper (for example `csrc/jit_kernels/impls/sm100_mxfp4_gemm_1d1d.hpp`).
   - Replace legacy assumptions such as `BLOCK_K == 128`, `SM100_MMA_MXF8F6F4_*`, and current SF cadence with MXFP4-specific constants.

4. Add low-level PTX wrappers and legality logic.
   - Mirror `deep_gemm/include/deep_gemm/common/sm100_utils.cuh:207-247` with MXFP4-specific 1SM / 2SM wrappers.
   - Audit `csrc/jit_kernels/heuristics/sm100.hpp:55-143`, `csrc/jit_kernels/impls/runtime_utils.hpp:95-235`, and `csrc/apis/layout.hpp:13-74` so block-K, TMA box sizes, swizzle, UTCCP staging, and packed UE8M0 transforms all match the MXFP4 route.

5. Extend tests before widening scope.
   - Add grouped FP4xFP4 generation to `tests/generators.py:41-77,152-165,221-252`.
   - Add correctness, regression, and dispatch-visibility assertions in `tests/test_fp8_fp4.py:60-157`.
   - Benchmark the new path against the current legacy path on the same eligible workloads before enabling default dispatch.

A concrete phase-1 dispatch sketch could look like:

```text
if use_mxfp4 == true
  and arch == 10
  and gemm_type == MGroupedContiguous
  and use_psum_layout == false
  and a.dtype == kPackedFP4
  and b.dtype == kPackedFP4
  and output == BF16
  and gran_k_a == 32
  and gran_k_b == 32
  and layout_is_supported_for_mxfp4(...)
  and k_is_supported_for_phase1(...):
    dispatch -> new MXFP4 grouped runtime
else if use_mxfp4 == true:
    hard error: unsupported MXFP4 grouped configuration
else:
    dispatch -> existing legacy grouped path
```

### Relevant References
- `csrc/apis/gemm.hpp:143-205` - current grouped-contiguous API entry point, SF transform call, and SM100 grouped runtime dispatch.
- `csrc/apis/layout.hpp:13-74` - SM100 scale-layout transform rules and `gran_k` handling for packed UE8M0 conversion.
- `deep_gemm/include/deep_gemm/common/types.hpp:5-14` - current `MmaKind` enum and element-size model; today it only distinguishes `BF16` vs `MXFP8FP4`.
- `csrc/jit_kernels/heuristics/common.hpp:152-183` - current config selection maps all non-BF16 cases to `MmaKind::MXFP8FP4` and fixes `block_k = 128`.
- `csrc/jit_kernels/heuristics/sm100.hpp:55-143` - SM100 SF/TMEM sizing and block legality currently keyed on the legacy `MmaKind` split.
- `csrc/jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp:18-214` - current SM100 JIT runtime generation and grouped-contiguous launch plumbing to mirror with distinct MXFP4 targets.
- `csrc/jit_kernels/impls/runtime_utils.hpp:50-235` - dtype stringification, packed-FP4 tensor-map encoding, and TMA descriptor construction constraints.
- `deep_gemm/include/deep_gemm/common/sm100_utils.cuh:207-247` - existing `mxf8f6f4.block_scale` inline PTX wrappers that the MXFP4 path must parallel.
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh:29-57,323-376` - legacy kernel assumptions that make a dedicated MXFP4 fork necessary (`BLOCK_K == 128`, legacy UTCCP cadence, legacy MMA wrapper selection).
- `deep_gemm/include/deep_gemm/common/scheduler.cuh:70-249` - shared grouped scheduler behavior for contiguous / masked / psum layouts that must be re-audited for MXFP4 block and CTA-group assumptions.
- `tests/generators.py:41-77,152-165,221-252` - current quant config coverage and missing grouped FP4xFP4 generation logic.
- `tests/test_fp8_fp4.py:60-157` - grouped correctness and perf harness to extend with MXFP4-specific validation.

## Dependencies and Sequence

### Milestones
1. **Lock the phase-1 contract and legality envelope**
   - Phase A: Decide the exact MXFP4 eligibility predicate: grouped type, layout(s), scale recipe, output type, K policy, and fallback/error behavior.
   - Phase B: Add `MmaKind::MXFP4`, distinct runtime/build names, and heuristic hooks so the host side can model MXFP4 legality without disturbing the legacy path.
2. **Add the dedicated MXFP4 runtime and kernel path**
   - Phase A: Implement MXFP4 low-level wrappers and a dedicated SM100 MXFP4 device kernel with its own block-K, UTCCP, and epilogue assumptions.
   - Phase B: Add a matching JIT/runtime wrapper and tensor-map setup, then route only eligible grouped-contiguous launches into it.
3. **Validate correctness, preserve compatibility, and gate rollout**
   - Phase A: Extend generators/tests for FP4xFP4 grouped correctness, plus dispatch-regression coverage proving legacy paths remain legacy.
   - Phase B: Benchmark the new path on eligible workloads, especially the user-relevant MoE-like grouped shape, and use those results to decide whether default dispatch is enabled immediately or kept gated.

Milestone 1 must land before kernel coding, because the kernel’s legality envelope determines block-K, tensor-map shapes, and dispatch behavior. Milestone 2 must land before validation, because tests need a stable runtime target to assert against. Milestone 3 closes the loop by proving correctness and limiting rollout risk before any broader scope expansion.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Add the MXFP4 phase-1 contract to the host stack: introduce an explicit MXFP4 opt-in parameter, define eligibility, hard-error policy for unsupported explicit-MXFP4 grouped cases, `MmaKind::MXFP4`, and distinct grouped runtime/build target names. | AC-1, AC-2 | coding | - |
| task2 | Audit packed-FP4 TMA, UTCCP, swizzle, and SF descriptor legality for grouped-contiguous `block_k = 256` so the kernel fork is based on validated constraints rather than copied legacy constants. | AC-2, AC-3 | analyze | task1 |
| task3 | Implement low-level `mxf4.block_scale` wrappers and a dedicated SM100 MXFP4 grouped 1D1D device kernel without modifying the existing SM100 grouped kernel and without requiring 2SM in phase 1. | AC-1, AC-3 | coding | task1, task2 |
| task4 | Add the new SM100 MXFP4 JIT/runtime wrapper and wire grouped-contiguous launch plumbing, tensor-map creation, and codegen keys to the new kernel. | AC-1, AC-3 | coding | task3 |
| task5 | Update grouped API dispatch so only eligible grouped-contiguous FP4xFP4 cases route to MXFP4, unsupported FP4xFP4 grouped cases hard-fail, and legacy non-MXFP4 grouped paths remain unchanged. | AC-2 | coding | task4 |
| task6 | Extend generators and tests with grouped FP4xFP4 correctness cases, legacy dispatch-regression cases, and unsupported-case hard-error assertions. | AC-4 | coding | task5 |
| task7 | Benchmark the new MXFP4 path against the legacy `mxf8f6f4` path on eligible workloads and report whether default dispatch should remain unconditional for all eligible phase-1 cases or require a follow-up tuning pass. | AC-4 | analyze | task6 |

## Claude-Codex Deliberation

### Agreements
- This work is not a local opcode substitution; it spans API dispatch, heuristics, JIT/runtime generation, device code, and tests.
- The current SM100 non-BF16 path is structurally tied to `MmaKind::MXFP8FP4` and `block_k = 128`, so MXFP4 needs its own legality model.
- Phase 1 should be narrow and explicitly gated, because the repo currently has no grouped FP4xFP4 validation coverage.

### Resolved Disagreements
- **Scope breadth**: The draft read like a broad grouped migration, but both Claude and Codex converged on a narrower first deliverable centered on grouped-contiguous FP4xFP4 plus explicit dispatch gating. This is the safest interpretation of the user request because it satisfies “new kernel + JIT dispatch” while limiting the risk of breaking existing grouped paths.
- **Rollout shape**: Instead of treating MXFP4 as an automatic replacement wherever FP4 appears, the converged direction is to add a separate runtime/build target and make dispatch eligibility explicit. This preserves backward compatibility and makes benchmark-driven rollout decisions possible.

### Convergence Status
- Final Status: `partially_converged`

## Pending User Decisions

- DEC-1: Phase-1 grouped scope
  - Claude Position: Ship `MGroupedContiguous` first and leave grouped-masked plus psum-layout support for a follow-up after the contiguous path is correct.
  - Codex Position: Same direction; keep phase 1 as a strict internal route instead of broadening to masked / psum immediately.
  - Tradeoff Summary: Narrow scope makes scheduler, epilogue, and TMA bugs easier to isolate. Broader scope reduces follow-up work but sharply increases the first-pass correctness and rollout risk.
  - Decision Status: `Phase 1 = MGroupedContiguous only`
- DEC-2: Behavior for FP4 inputs that are not MXFP4-eligible
  - Claude Position: Keep the public API stable and fall back to the existing `mxf8f6f4` grouped path unless a case is known to be semantically unsafe.
  - Codex Position: Same direction; use a strict MXFP4 route only for explicitly eligible cases and keep all others on the legacy path.
  - Tradeoff Summary: Fallback maximizes compatibility and preserves current callers. Hard errors make unsupported cases obvious but can break workflows that currently succeed on the legacy kernel.
  - Decision Status: `Only explicit MXFP4 opt-in may use the new path; unsupported opt-in cases hard-error; calls without opt-in use the legacy path`
- DEC-3: Whether 2SM is required in the first MXFP4 delivery
  - Claude Position: Do not make 2SM mandatory for initial acceptance; a 1SM-first MXFP4 path is acceptable if it substantially reduces correctness/debug risk.
  - Codex Position: Split the work into separate deliverables: 1SM first, then 2SM once correctness and legality are stable.
  - Tradeoff Summary: 2SM may matter for peak throughput, but it couples peer-CTA sync, per-SM tile ownership, and epilogue partitioning into the riskiest part of the migration.
  - Decision Status: `Phase 1 does not require 2SM`
- DEC-4: Default-dispatch rollout gate
  - Claude Position: Enable default MXFP4 dispatch only after benchmark parity or improvement is demonstrated on eligible workloads, including `num_groups=32, M=32768, N=2048, K=7168`.
  - Codex Position: Start with a debug/env-gated rollout and switch to default dispatch only after the benchmark bar is met.
  - Tradeoff Summary: Immediate default routing simplifies usage but risks silent regressions. Gated rollout keeps the new path observable while profiling and tuning finish.
  - Decision Status: `Resolved by explicit opt-in parameter; without opt-in, always use the legacy path`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Additional Constraints for This Change
- Preserve the current SM100 grouped legacy kernel and its runtime names as compatibility anchors.
- Prefer MXFP4-specific identifiers (`mxfp4` / `mxf4`) for new runtime, kernel, and helper names so dispatch traces are unambiguous.
- Keep hot-path observability lightweight; prefer existing debug/env mechanisms or deterministic runtime naming over ad-hoc logging in the kernel body.

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

如果你是把 **B200 上 DeepGEMM 的 grouped GEMM** 从现在这条 **`mxf8f6f4.block_scale`** 路，改成 **`mxf4.block_scale`**，我会按下面这个顺序看。
先给一句总判断：

> **这不是“把 operand dtype 从 mxf8 改成 mxf4”这么简单，而是要同时改 layout、tile-K、copy/pack、scale 索引、1SM/2SM 分工和 epilogue。**
> 真正最容易出错的，通常不是 MMA 指令本身，而是 **前后的数据组织链路**。 ([NVIDIA Docs][1])

## 1) 先确认：`mxf4` 不是 `mxf8f6f4` 的直接替身

`mxf4` 在 SM100 上只覆盖 **`mx_float4_t x mx_float4_t`**，并且 **只支持 TN layout**；而 `mxf8f6f4` 覆盖 `{mx_float4,mx_float6,mx_float8}` 的多种组合，并支持 **TN / NT / NN / TT**。对 `mx_float4_t x mx_float4_t` 这组输入，`mxf4` dense 的 A/B 对齐要求是 **32 / 32 elements**，而 `mxf8f6f4` dense 这一路是 **128 / 128 elements**。DeepGEMM 的公开 SM100 接口虽然支持四种布局，但输入转置/打包需要用户自己做；所以如果你现在的 grouped kernel 是 `*_nt` 或 `*_nn`，切到 `mxf4` 的第一件事不是改 MMA opcode，而是先决定：**改接口到 TN，还是在前面加一次预转置/重打包**。([NVIDIA Docs][1])

这点很关键，因为很多人会以为：

* 现在是 `mx_float4 x mx_float4`
* 那就把 `mxf8f6f4` 换成 `mxf4` 就行

但实际不行。
**如果你现在线路依赖 NT/NN/TT 中任意一种，`mxf4` 就不是 drop-in replacement。** ([NVIDIA Docs][1])

## 2) 量化和 scale 路也要一起改，不只是 data tensor

如果你走的是 **`mxf4`**（不是 `mxf4nvf4`），那 scale dtype 还是 **`ue8m0`**，dense 的 scale vector size 还是 **32**，sparse 是 **64**。这点和 DeepGEMM 在 SM100 上公开使用的 **packed UE8M0** 约定是兼容的，所以 **scale 的基本 dtype/pack 方向可以继续沿用**。但要注意：`mxf4` 强制 A、B 两边都变成 **`mx_float4_t`**；不像 `mxf8f6f4` 那样还能做 `mx_float4 x mx_float8`、`mx_float6 x mx_float8` 之类的混搭。所以你不只是换 kernel，还得把 **两边的 quantizer / calibration / clipping 策略** 一起改成 4-bit 版本。([NVIDIA Docs][1])

如果你其实想走的是 **`mxf4nvf4`**，那事情会再变一层：
scale dtype 会从 `ue8m0` 变成 **`ue4m3`**，dense/sparse 的 scale vector size 也会从 `32/64` 变成 **`16/32`**。这个分支和纯 `mxf4` 不是一回事，别把两条线混在一起。([NVIDIA Docs][1])

## 3) `mxf4` 最大的结构变化：**tile-K 变了**

对 `mx_float4_t x mx_float4_t` 这组输入，SM100 上：

* `mxf8f6f4` dense 的合法 MMA tile shape 是 **`...x128`**
* `mxf4` dense 的合法 MMA tile shape 是 **`...x256`**。([NVIDIA Docs][1])

这意味着你现在 grouped GEMM 的 mainloop 如果是围着 `K_tile = 128` 写的，那么切到 `mxf4` 后，至少这些东西要重看：

* stage 数量
* smem / TMEM 占用
* scale 的 K 维步进
* TMA / descriptor 的 K extent
* K tail / remainder 逻辑
* config heuristic 里对 K 的偏好

因为这不是“同样的 tile 里换更小的数据”，而是 **MMA 原子的 K 覆盖范围变大了**。CUTLASS 文档还明确说 `MMA_TileShape_K` 通常约等于 `4 * instruction-K`，所以你可以把 `mxf4` 理解成 **更深 K 的 MMA**；你当前如果有一堆针对 `x128` 调出来的 grouped 配置，换过去后通常都得重新调。([NVIDIA Docs][1])

对 grouped GEMM 来说，这一点尤其重要：
因为 grouped 场景常常不是“大而整”的单一 shape，而是很多 group 共享固定 N/K、只有 M 在变。`K_tile` 从 128 变 256 后，**短 K、非整除 K、或者你当前依赖的尾块处理**，很容易突然成为 correctness 或 utilization 的问题。([GitHub][2])

## 4) 最容易踩坑的其实是 copy / pack / TMA，不是 MMA 本体

PTX 文档明确写了，`tcgen05.cp` 可以把数据从 shared memory 异步搬到 Tensor Memory，而且支持 **4-bit → 8-bit 的可选解压**。这意味着 mxf4 路线很可能会涉及一种和你现有 FP8 不同的 **sub-byte pack + copy** 组织方式。([NVIDIA Docs][3])

更麻烦的是，4-bit sub-byte tensor copy 对 `.b4x16_p64` 这类格式有一串很硬的限制：

* `Box-Size[0]` 必须是 **64B**
* `Tensor-Size[0]` 必须是 **64B 的倍数**
* tensorCoords 的第一维必须是 **128 的倍数**
* global 地址和各维 stride 必须 **32B 对齐**
* swizzle 也有限制。([NVIDIA Docs][3])

所以如果你现在 DeepGEMM 这条 `mxf8f6f4` grouped kernel 的前端 pack/TMA 描述符是围着 FP8 写的，**不要假设它能直接复用到 mxf4**。
我会把下面几样单独拿出来重查一遍：

* A/B 的 GMEM pack format
* TMA tensor-map 的 box size / tensor size / stride
* shared 中的 padding
* shared → TMEM 的 copy shape
* swizzle 是否仍合法
* scale tensor 的 index/descriptor 是否跟着新的 K_tile 走

这一步没改对，通常会表现成两种症状：
**要么结果错，要么 profile 上 `utccp` / shared traffic / replay 突然变坏。**

## 5) 1SM / 2SM 不是“后面再说”，一开始就要决定

`mxf4` 和 `mxf8f6f4` 都支持 1SM / 2SM 的 `tcgen05.mma`，但 PTX 要求一个 kernel 里所有 `tcgen05` 指令——包括 `alloc / cp / shift / mma / commit`——都必须使用同一个 `cta_group`。也就是说，你一旦决定走 2SM，就不是只把 `mma` 换成 2SM；**整条 tcgen05 管线都要跟着变成 2SM。** ([NVIDIA Docs][3])

对 grouped GEMM，这一点影响很实际。
CUTLASS 文档给了一个很清楚的例子：如果是 2SM 的 `MmaTileShape_MNK = 256x256x128`，那每个 SM 真正负责的 `PerSmTileShape_MNK` 是 **`128x256x128`**，也就是 **M 方向一分为二**。你前面自己也已经看到 2SM 时 B operand fetch 会变，这和 2SM 的数据/输出分工是一致的。([NVIDIA Docs][1])

所以你改 grouped kernel 时，至少要一起改这些：

* group offset 到 per-SM tile 的映射
* peer CTA 的同步 / cp / commit
* epilogue 的输出区间划分
* 如果有 mask/contiguous expert slicing，是否还和 2SM 的 per-SM tile 对齐

否则经常会出现“主循环对了，但 epilogue 写回错了/重叠了/漏了”的问题。([NVIDIA Docs][3])

## 6) accum 不用猜：block-scaled `tcgen05.mma` 的累加还是 float

这一点反而简单：
CUTLASS 官方示例直接写了，block-scaled `tcgen05.mma` 的 `ElementAccumulator` **always float**。所以你切 `mxf4` 时，不需要把 accum buffer 逻辑改成 bf16/f16；TMEM 里的 D/accum 这条线仍按 **FP32 accum** 理解。output dtype 可以在 epilogue 再下变换，但 **内部累加不是 mxf4**。([NVIDIA Docs][1])

## 7) 性能上别预设“算力翻倍，kernel 就翻倍”

官方文档里，SM100 的 `mxf4` 吞吐被标成 **4x Hopper FP8 Tensor Core**，而 `mxf8f6f4` 是 **2x Hopper FP8 Tensor Core**。所以**从 backend 算术能力**看，`mxf4` 的确更强。([NVIDIA Docs][1])

但结合你前面那组 profile，我不会把预期放成“kernel 直接接近 2x”。
因为你现在的瓶颈已经更像是 **shared / TC operand feed，尤其 B operand 这条链**。这种情况下，切到 `mxf4` 后很可能发生的是：

* backend 更强了
* 但 `tcgen05.cp` / `utccp` / shared→TMEM / B operand feed 变得更贵
* 最后只是把瓶颈从 `mxf8f6f4` 的某一段，迁移到 `mxf4` 的 copy/feed 段

所以迁移后我最先盯的不是 TFLOPS，而是这几类指标：

* `utccp` 相关指令数/bytes
* `utcmma_matrix_b_scope_*`
* `l1tex__data_pipe_tc_wavefronts*`
* `smsp__average_warps_issue_stalled_long_scoreboard*`
* shared bank/wavefront 指标

如果这些变差，说明你不是“算得慢”，而是“喂得更难了”。

## 8) 我建议的落地顺序

如果是我来改，我不会一上来就把整套 grouped path 全切掉，而是按这个顺序：

1. **先做 dense + TN-only + `mx_float4 x mx_float4` + 2SM**
2. 让 **K 先只跑 256 的整数倍**
3. scale 先继续用 **UE8M0**
4. 先验证数值和主循环 profile
5. 再接 grouped contiguous 的 offset / mask / tail
6. 最后再补非整除 K、短 K 和其他 layout 适配

这样你能尽快把问题收敛到：

* 是 layout 不对
* 是 copy path 不对
* 还是 2SM/epilogue 分工不对

而不是一开始把所有变量都混在一起。

一句话总结：

> **把 DeepGEMM grouped GEMM 从 `mxf8f6f4` 切到 `mxf4`，最该小心的不是 MMA 指令名，而是 TN-only 约束、A/B 双边都要变成 mx4、tile-K 从 128 变 256、4-bit copy/TMA 的对齐与 padding、以及 2SM 下 per-SM tile/epilogue 的重新分工。**
> **如果这些没一起改，通常不是“有点慢”，而是直接“不对”或者“热点跑偏”。** ([NVIDIA Docs][1])

你要是愿意，我下一条可以直接按 **DeepGEMM grouped-contiguous** 的语境，给你列一个更工程化的迁移 checklist：
**“哪些 template / descriptor / scale packer / epilogue 索引需要改”**。

[1]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html "Blackwell SM100 GEMMs — NVIDIA CUTLASS Documentation"
[2]: https://github.com/deepseek-ai/DeepGEMM "GitHub - deepseek-ai/DeepGEMM: DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling · GitHub"
[3]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=async "1. Introduction — PTX ISA 9.2 documentation"

这是我用chatgpt询问的关于把这个kernel里的mxf8f6f4，写成用mxfp4 mma指令替换时，需要注意的事项。我现在想让你具体来做这件事情。不过不要去修改原始kernel，新增一个新的，专门针对mxfp4的kernel，jit时dispatch到这里。你先阅读下相关文件，确保你理解了，然后生成 plan。
--- Original Design Draft End ---
