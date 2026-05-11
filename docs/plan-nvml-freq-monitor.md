# <Plan Title>

## Goal Description
<Clear, direct description of what needs to be accomplished>

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: <First criterion>
  - Positive Tests (expected to PASS):
    - <Test case that should succeed when criterion is met>
    - <Another success case>
  - Negative Tests (expected to FAIL):
    - <Test case that should fail/be rejected when working correctly>
    - <Another failure/rejection case>
  - AC-1.1: <Sub-criterion if needed>
    - Positive: <...>
    - Negative: <...>
- AC-2: <Second criterion>
  - Positive Tests: <...>
  - Negative Tests: <...>
...

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
<Affirmative description of the most comprehensive acceptable implementation>
<This represents completing the goal without over-engineering>
Example: "The implementation includes X, Y, and Z features with full test coverage"

### Lower Bound (Minimum Acceptable Scope)
<Affirmative description of the minimum viable implementation>
<This represents the least effort that still satisfies all acceptance criteria>
Example: "The implementation includes core feature X with basic validation"

### Allowed Choices
<Options that are acceptable for implementation decisions>
- Can use: <technologies, approaches, patterns that are allowed>
- Cannot use: <technologies, approaches, patterns that are prohibited>

> **Note on Deterministic Designs**: If the draft specifies a highly deterministic design with no choices (e.g., "must use JSON format", "must use algorithm X"), then the path boundaries should reflect this narrow constraint. In such cases, upper and lower bounds may converge to the same point, and "Allowed Choices" should explicitly state that the choice is fixed per the draft specification.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
<Text description, pseudocode, or diagrams showing ONE possible implementation path>

### Relevant References
<Code paths and concepts that might be useful>
- <path/to/relevant/component> - <brief description>

## Dependencies and Sequence

### Milestones
1. <Milestone 1>: <Description>
   - Phase A: <...>
   - Phase B: <...>
2. <Milestone 2>: <Description>
   - Step 1: <...>
   - Step 2: <...>

<Describe relative dependencies between components, not time estimates>

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | <...> | AC-1 | coding | - |
| task2 | <...> | AC-2 | analyze | task1 |

## Claude-Codex Deliberation

### Agreements
- <Point both sides agree on>

### Resolved Disagreements
- <Topic>: Claude vs Codex summary, chosen resolution, and rationale

### Convergence Status
- Final Status: `converged` or `partially_converged`

## Pending User Decisions

- DEC-1: <Decision topic>
  - Claude Position: <...>
  - Codex Position: <...>
  - Tradeoff Summary: <...>
  - Decision Status: `PENDING` or `<User's final decision>`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

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

# Draft: 用 NVML API 精细量化降频情况及其原因

## 背景

DeepGEMM mega_moe kernel 在 B200 上运行时，NVLink 通信流量引发 DVFS 降频，导致 SM 频率从 1965 MHz 最大 boost 降至 ~1350 MHz（大 batch 时）。之前的实验（047/048）通过 kernel 内 `clock64() / globaltimer` profiler 测得了 per-SM 平均频率，但这只能测到时间区间内的平均值，无法观测频率随时间变化的动态行为。

## 目标

利用 NVML API 在 kernel 运行期间从 host 侧监控 GPU 状态，补充 kernel 内 profiler 无法覆盖的维度：

1. **Power violation 累计时间**：通过 `nvmlDeviceGetViolationStatus(POWER)` 的增量，量化每次 kernel 运行期间有多少时间受到功耗墙限制
2. **Throttle reason 分类**：通过 `nvmlDeviceGetCurrentClocksThrottleReasons` 区分 power cap、thermal、HW slowdown 等不同降频原因
3. **实时功耗曲线**：通过 `nvmlDeviceGetPowerUsage` 轮询，画出 kernel 运行期间的功耗时间序列
4. **SM clock 轮询**（辅助）：`nvmlDeviceGetClockInfo(SM)` 虽然分辨率有限（驱动缓存更新周期 ~10-100ms），但对 B32k 等长 kernel（~10ms）仍可提供粗粒度参考

## 已验证的 API 可用性

在 B200 (driver 590.48.01) 上已确认：
- `nvmlDeviceGetClockInfo(SM)`: OK, 查询延迟 1-50 us
- `nvmlDeviceGetViolationStatus(POWER)`: OK, 累计 violation 时间可用
- `nvmlDeviceGetCurrentClocksThrottleReasons`: OK, 支持全部 9 个 reason bit
- `nvmlDeviceGetPowerUsage`: OK, mW 精度
- `nvmlDeviceGetSamples(TOTAL_POWER)`: OK, ~20ms 间隔硬件采样
- `nvmlDeviceGetSamples(PROCESSOR_CLK)`: **Not Supported** on B200
- `nvidia-smi -lgc MIN,MAX`: OK, 可用于锁频对照实验
- `nvmlDeviceGetSupportedGraphicsClocks`: OK, 247 档位 120-1965 MHz

## 已知限制

- NVML 是"缓存读"，驱动内部更新周期 ~10-100ms，对 2-3ms 的短 kernel 测不准
- `PROCESSOR_CLK` 硬件采样 buffer 在 B200 上不可用
- kernel 内 `clock64()` 仍是最精确的频率测量手段（per-SM、几 ns 精度），NVML 作为补充而非替代

## 实验设计

### 实验 A：Power violation 增量 × 频率散点

跑 N 次 kernel（如 200 次），每次记录：
- NVML power violation 时间增量（kernel 前后各读一次 ViolationStatus）
- Kernel wall time（CUDA event）
- Kernel 平均 SM clock（已有的 clock64() profiler）

画散点图：横轴 power-violation 占比，纵轴平均 SM clock。

### 实验 B：锁频对照

- 锁频（`nvidia-smi -lgc 1965,1965`）重跑，拿到无降频基线 f₀ 和 t₀
- 自由运行时的 f̄ 和 t̄
- 性能损失 = (t̄ - t₀) / t₀
- 降频幅度 = (f₀ - f̄) / f₀
- 频率敏感度 = 性能损失 / 降频幅度

### 实验 C：Throttle reason 分类

在 kernel 运行期间高频轮询 throttle reasons，统计各 reason 的出现频率和时间占比。

### 实验 D：功耗时间序列

用 `nvmlDeviceGetPowerUsage` 或 `nvmlDeviceGetSamples(TOTAL_POWER)` 采集功耗曲线，与 kernel 启停时间对齐。

## 与现有 profiler 的关系

当前 kernel 内 profiler（`clock64() + globaltimer`）已经提供了 per-SM 级别的平均频率和 compute utilization。NVML 监控的价值在于：
- 提供 **host 侧** 的独立验证
- 量化 **power violation 时间占比**（kernel 内无法测量）
- 区分 **降频原因**（power vs thermal vs other）
- 提供 **功耗绝对值**（kernel 内无法测量）

## 实现形态

一个独立的 Python 测试/bench 脚本，可以：
- 复用现有的 `test_mega_moe.py` 中的 kernel 调用逻辑
- 在 kernel 前后/期间调用 NVML API
- 输出结构化数据（CSV/JSON）
- 可选：画图输出

--- Original Design Draft End ---
