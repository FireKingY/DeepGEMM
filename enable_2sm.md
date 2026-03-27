# Enable 2SM MMA For SM100 MGroupedContiguous

## Goal Description

为 DeepGEMM 的 SM100 grouped GEMM 增加一个显式 opt-in 的 2SM MMA 开关，但本次变更只覆盖 `MGroupedContiguous`。当调用方显式请求 2SM 时，host 侧必须先验证每个 group 的 padded `M` 都满足 `padded_m % 256 == 0`；若任一 group 不满足，或调用落在本计划未覆盖的路径上，则直接报错，不允许静默回退到 1SM。未显式请求 2SM 时，现有行为必须保持不变。

本计划的最小实现范围聚焦于 SM100 FP8/FP4 grouped contiguous 1D1D 路径；是否把同样的开关和校验复用到 SM100 BF16 grouped contiguous，作为可选扩展，不阻塞本次交付。

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: 为 `MGroupedContiguous` 新增显式 2SM 参数，并将其完整传递到 SM100 grouped contiguous 调度路径
  - Positive Tests (expected to PASS):
    - `m_grouped_fp8_fp4_gemm_nt_contiguous(..., enable_2sm_mma=False)` 与当前主线行为一致，数值结果与现有测试基线一致
    - `m_grouped_fp8_fp4_gemm_nt_contiguous(..., enable_2sm_mma=True)` 在合法规则 shape 上可以走到 SM100 grouped contiguous dispatch
    - `m_grouped_fp8_fp4_gemm_nn_contiguous(..., enable_2sm_mma=True)` 作为别名入口也能正确透传参数
  - Negative Tests (expected to FAIL):
    - 在 `MGroupedMasked` 或 `use_psum_layout=True` 的调用上显式传入 `enable_2sm_mma=True` 时，接口直接报错并说明当前只支持 `MGroupedContiguous`
    - 在非 SM100 路径或未实现的 grouped 变体上显式传入 `enable_2sm_mma=True` 时，接口直接报错，而不是静默忽略参数

- AC-2: 当 `enable_2sm_mma=True` 时，host 侧必须做确定性的 shape legality 检查
  - Positive Tests (expected to PASS):
    - 构造每个 group 的 padded `M` 分别为 `256`, `512`, `768` 的 `MGroupedContiguous` case，校验通过并继续执行
    - 对 no-psum contiguous layout，若所有 group 的 padded `M` 都满足 `padded_m % 256 == 0`，则校验函数返回成功
  - Negative Tests (expected to FAIL):
    - 任意一个 group 的 padded `M` 为 `128`, `384`, `640` 等非 `256` 倍数时，显式 2SM 请求直接报错
    - `grouped_layout` 无法解析出合法 contiguous group 边界，或存在与 `num_groups` 不一致的异常布局时，显式 2SM 请求直接报错
  - AC-2.1: legality 检查必须与当前实现约束保持一致，而不是依赖运行时“碰巧正确”
    - Positive:
      - 校验逻辑明确基于当前 `MGroupedContiguous` 的 `BLOCK_M=128` 约束，要求每个 group 的 M-tile 数为偶数
      - 错误信息明确指出失败原因来自 “group padded M not divisible by 256”
    - Negative:
      - 不能仅依赖 `m % 512 == 0`、`num_groups == 1` 或其它全局条件替代逐 group 校验
      - 不能把不满足规则的 shape 自动回退为 1SM 并继续执行

- AC-3: 当 `enable_2sm_mma=True` 且 shape 合法时，配置选择必须显式请求并实际得到 `kNumMulticast=2`
  - Positive Tests (expected to PASS):
    - 配置构建阶段能显式请求 `MulticastConfig{2, false}`，并在 grouped contiguous 有效 shape 上返回 2SM 配置
    - launch 参数中的 cluster size 为 `2`，且最终 kernel 配置满足 `num_multicast == 2`、`is_multicast_on_a == false`
    - 至少一个集成测试在合法 shape 上开启显式 2SM 后数值正确
  - Negative Tests (expected to FAIL):
    - 如果显式 2SM 请求发出后，配置层仍返回 `num_multicast == 1`，则 host 侧直接报错
    - 不能通过“只改 heuristics 默认值”来隐式开启 2SM；显式参数必须是唯一开关来源

- AC-4: 默认路径保持兼容，显式 2SM 只影响被请求的 `MGroupedContiguous`
  - Positive Tests (expected to PASS):
    - 现有 grouped contiguous 测试在不传或关闭 `enable_2sm_mma` 时继续通过
    - 不规则 shape 在默认 1SM 路径下继续按现有行为运行，不新增错误
  - Negative Tests (expected to FAIL):
    - 不能因为新增显式 2SM 参数而改变默认 heuristic 下的 block size、multicast 选择或已有 API 语义
    - 不能把默认 grouped contiguous 自动升级为 2SM

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

实现一个可复用的“显式 2SM 请求 + grouped legality 校验 + 强制 2SM 配置”框架，先在 SM100 FP8/FP4 `MGroupedContiguous` 落地，并在不扩大语义风险的前提下，把同样的 host-side 校验与 opt-in 接口复用到 SM100 BF16 `MGroupedContiguous`。错误路径具备清晰报错信息，测试覆盖合法/非法 shape、默认路径回归、别名入口和 dispatch 行为。

### Lower Bound (Minimum Acceptable Scope)

只实现 SM100 FP8/FP4 `MGroupedContiguous` 1D1D 路径的显式 2SM 开关与 legality 校验；`enable_2sm_mma=True` 时只支持 no-psum contiguous layout，并要求每个 group 的 padded `M` 满足 `padded_m % 256 == 0`。shape 不满足或路径不在支持范围内时直接报错；参数未开启时完全保持现状。

### Allowed Choices

> **Note on Deterministic Designs**: 本需求属于强约束设计。`MGroupedContiguous`、显式 opt-in、逐 group `padded_m % 256 == 0`、非法时直接报错，这些选择都是固定的，不允许改成启发式或静默回退。

- Can use: 新增 host-side 校验 helper；在 config 选择层增加显式 multicast override；在 API 层增加可选布尔参数；复用现有 grouped contiguous tests 并补充定制 case
- Cannot use: 通过修改默认 heuristics 自动开启 2SM；支持 `MGroupedMasked`；在非法 shape 上自动 fallback；依赖 device-side 动态保护来掩盖 grouped pairing 风险

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

一种可行实现路径如下：

1. 在 grouped contiguous API 增加显式参数，例如 `enable_2sm_mma`，默认 `false`。
2. 当参数为 `false` 时，完全沿用当前流程。
3. 当参数为 `true` 时，先做 host-side legality 检查：
   - 仅允许 `MGroupedContiguous`
   - 仅允许 no-psum contiguous layout
   - 扫描 `grouped_layout`，恢复每个 group 的 padded span，要求每个 span 满足 `padded_m % 256 == 0`
4. legality 通过后，配置构建阶段显式请求 `MulticastConfig{2, false}`，并在最终配置上断言确实拿到了 2SM。
5. dispatch 与 kernel 主体尽量少改，只在 host/config 层保证 2SM 只会出现在被证明安全的 grouped contiguous shape 上。

### Relevant References

- `csrc/apis/gemm.hpp` - grouped contiguous FP8/FP4 与 BF16 API 入口，适合新增显式参数与前置校验
- `csrc/jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp` - SM100 grouped contiguous FP8/FP4 dispatch，适合接入显式 2SM 请求
- `csrc/jit_kernels/heuristics/common.hpp` - `get_best_config` 与 `MulticastConfig` 选择逻辑
- `csrc/jit_kernels/heuristics/sm100.hpp` - 当前显式禁止 `MGrouped*` 使用 multicast 的架构策略
- `deep_gemm/include/deep_gemm/common/scheduler.cuh` - grouped tile pairing 语义，说明为什么必须逐 group 保证偶数 M-tile
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh` - SM100 2CTA UMMA 主路径
- `tests/test_fp8_fp4.py` - grouped contiguous 现有测试入口，适合补充显式 2SM case

## Dependencies and Sequence

### Milestones

1. Milestone 1: 固化接口与 legality 约束
   - Phase A: 明确显式参数的 API 位置、默认值和报错语义
   - Phase B: 实现 host-side legality helper，能从 `grouped_layout` 恢复每组 padded `M`
2. Milestone 2: 显式 2SM 配置接入
   - Phase A: 在 grouped contiguous dispatch 中接入 legality check 和显式 multicast 请求
   - Phase B: 在 config 层支持“显式请求 2SM，不走默认 grouped 禁用策略”
3. Milestone 3: 验证与回归
   - Phase A: 补充合法/非法 shape 的单测与集成测试
   - Phase B: 验证默认 1SM 路径无回归，显式 2SM 路径数值正确

依赖关系上，Milestone 1 先于 Milestone 2；只有当 legality helper 能稳定区分合法/非法 grouped contiguous shape 后，才应开放显式 2SM 配置。Milestone 3 依赖前两者全部完成。

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 复核 `MGroupedContiguous` 当前 `BLOCK_M=128`、2CTA peer pairing 与 `padded_m % 256 == 0` 的对应关系，并形成实现备注 | AC-2 | analyze | - |
| task2 | 在 grouped contiguous API 增加 `enable_2sm_mma` 参数，并定义默认行为与报错语义 | AC-1, AC-4 | coding | task1 |
| task3 | 实现 host-side legality helper，逐 group 解析 contiguous `grouped_layout` 并检查 `padded_m % 256 == 0` | AC-2 | coding | task2 |
| task4 | 在 SM100 grouped contiguous dispatch / config 选择层增加显式 2SM 请求与结果断言 | AC-3 | coding | task3 |
| task5 | 为合法/非法 shape、别名入口、默认路径回归补充测试 | AC-1, AC-2, AC-3, AC-4 | coding | task4 |
| task6 | 评估同一套 opt-in + legality 结构是否可以无额外语义风险地复用于 SM100 BF16 grouped contiguous | AC-4 | analyze | task4 |

## Claude-Codex Deliberation

### Agreements

- 2SM 不能通过改默认 heuristics 直接打开，必须是显式 opt-in
- 本次范围只应覆盖 `MGroupedContiguous`，不应把 `MGroupedMasked` 混入同一交付
- legality 必须在 host 侧完成，且要逐 group 检查，而不是看全局 `m`
- 显式请求 2SM 但 shape 不合法时，必须报错，不能静默回退

### Resolved Disagreements

- 范围宽度: 为降低风险，主交付以 SM100 FP8/FP4 `MGroupedContiguous` 为下界；SM100 BF16 只作为可选上界，不阻塞主实现。这样可以先验证最直接的 grouped 2SM 需求，再决定是否做共享扩展。

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- 无。当前计划按用户已明确给出的约束执行：只开 `MGroupedContiguous`，要求每组 padded `M % 256 == 0`，通过显式参数请求 2SM，shape 不满足时直接报错。

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or similar workflow markers
- 代码中的参数名、helper 名和错误信息应直接表达域含义，例如 “explicit 2SM request”, “group padded M must be divisible by 256”
- 不要在 kernel device 代码里加入与计划结构绑定的术语；计划术语只保留在文档里
