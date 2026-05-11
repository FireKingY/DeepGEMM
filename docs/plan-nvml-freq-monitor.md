# NVML Host 侧频率降频监控 — Mega MoE Kernel

## Goal Description

在现有 `tests/test_mega_moe.py` 中增加 NVML 监控功能，利用 NVML API 在 B200 GPU 上运行 Mega MoE kernel 时，监控 GPU 频率降频、功耗和降频原因。补充现有 kernel 内 `clock64()/globaltimer` profiler 无法覆盖的 host 侧维度：power violation 时间、降频原因分类、功耗曲线和 SM 时钟动态。不新建独立脚本，直接在 `test_mega_moe.py` 中通过命令行参数（如 `--nvml-monitor`）启用 NVML 监控。

脚本实现四个实验：
- **实验 A**：Power violation 增量 × kernel 频率散点图（N 次迭代）
- **实验 B**：锁频 vs 自由运行性能对比
- **实验 C**：后台线程轮询 throttle reason 时间序列分类
- **实验 D**：通过 `nvmlDeviceGetSamples(TOTAL_POWER)` 硬件缓冲区获取功耗时间序列

## Acceptance Criteria

- AC-1: 每次 kernel 调用的 power violation 测量
  - 正向测试：
    - B32k（大 batch）运行 N 次迭代，产生 N 个 violation 增量值，大部分 > 0（功耗受限区间）
    - B512（小 batch）运行 N 次迭代，产生 N 个 violation 增量值，大部分 ≈ 0（无功耗限制）
    - Violation 增量始终非负（累计计数器单调递增）
  - 反向测试：
    - 未安装 pynvml 时，脚本以清晰的错误信息退出而非崩溃
    - NVML 初始化失败时（如 GPU 索引错误），脚本报告失败原因

- AC-2: 后台轮询线程的 throttle reason 分类
  - 正向测试：
    - 轮询线程尽可能高频轮询（~1ms 间隔，受 NVML 缓存更新周期限制）；由于单次 kernel 运行时间极短（几百到 ~3000us），有效 throttle reason 数据来自跨多次 kernel 调用的连续监控
    - 输出包含每个 reason bit 的时间占比（如 `SW_POWER_CAP: 85%`、`HW_SLOWDOWN: 12%`）
    - 轮询线程在 kernel 启动前开始、完成后停止
  - 反向测试：
    - kernel 提前结束时，轮询线程不会挂起或死锁
    - 测量窗口关闭后，轮询线程不继续运行

- AC-3: 硬件采样缓冲区的功耗时间序列
  - 正向测试：
    - `nvmlDeviceGetSamples(TOTAL_POWER)` 获取覆盖 kernel 执行窗口的硬件缓冲功耗样本
    - 功耗样本包含与 kernel 启停对齐的时间戳（CUDA event 时间）
    - 输出功耗值（瓦特），每 20ms kernel 时长至少 1 个样本
  - 反向测试：
    - 硬件采样缓冲区为空时（kernel 太短），脚本报告"样本不足"而非失败

- AC-4: 后台线程 SM 时钟轮询
  - 正向测试：
    - 后台线程在 kernel 执行期间采集 `nvmlDeviceGetClockInfo(SM)` 读数
    - B32k kernel（~3ms / ~3000us）至少捕获 1 个时钟读数
    - 时钟值为 MHz，在 GPU 支持范围内（120-1965 MHz）
  - 反向测试：
    - 对于极短 kernel（B512, ~2ms），脚本说明时间分辨率不足，而非报告误导性数据

- AC-5: 锁频对照实验（实验 B）
  - 正向测试：
    - 脚本支持"锁频"模式（使用 `nvidia-smi -lgc`）和"自由"模式
    - 计算并报告：性能损失 = (t_free - t_locked) / t_locked
    - 计算并报告：降频幅度 = (f_locked - f_free) / f_locked
    - 计算并报告：频率敏感度 = 性能损失 / 降频幅度
  - 反向测试：
    - `nvidia-smi -lgc` 失败时（权限不足等），脚本报告错误并建议用 sudo 或跳过实验 B

- AC-6: 结构化数据输出
  - 正向测试：
    - 每次迭代结果写入 CSV，列包括：iteration, wall_time_us, avg_sm_mhz, violation_delta_ns, power_watts
    - Throttle reason 时间占比写入汇总区域
    - 功耗时间序列写入单独的 CSV
  - 反向测试：
    - 脚本不会静默覆盖已有输出文件

- AC-7: 多 rank 兼容性
  - 正向测试：
    - 脚本在 EP=8 分布式环境下运行（8 GPU，8 进程）
    - 每个 rank 通过 `nvmlDeviceGetHandleByIndex(local_rank)` 监控自己的 GPU
    - 输出文件按 rank 分离（如 `nvml_rank0.csv`、`nvml_rank1.csv`）
  - 反向测试：
    - 未初始化 torch.distributed 时，脚本回退到单 GPU 模式

- AC-8: 与现有 kernel profiler 集成
  - 正向测试：
    - 脚本复用 `test_mega_moe.py` 的 kernel 调用逻辑（权重、缓冲区、激活参数）
    - 可选传入 `profiler_buffer` 以同时采集 kernel 内频率
    - 两者同时存在时，输出包含 kernel 内中位 MHz 和 NVML 报告的 SM 时钟，用于交叉验证
    - 必须显式 cross-check：对比 NVML 报告的 SM 时钟与 kernel 内 clock64()+globaltimer 测得的频率，确认两者趋势一致（如大 batch 降频、小 batch 不降频），并报告偏差幅度
  - 反向测试：
    - 不传 profiler_buffer 时脚本正常工作（纯 NVML 模式）

## Path Boundaries

### 上界（最大可接受范围）
实现包含全部四个实验（A-D）、后台轮询线程（clock/throttle）、硬件功耗采样、锁频对照自动化、按 rank CSV 输出、可选 matplotlib 散点图/时间序列图、可配置迭代次数和 batch 大小、以及与 kernel 内 profiler 的交叉验证。

### 下界（最小可接受范围）
实现包含实验 A 和 B（前后快照 + 锁频对照）、CSV 输出、多 rank 支持、可配置 batch 大小和迭代次数。实验 C/D（后台轮询线程）功能可用但输出格式可以更简单。

### 允许的选择
- 可用：`pynvml`（nvidia-ml-py）做 NVML 绑定、`threading.Thread` 后台轮询、`csv` 模块输出、`matplotlib` 可选绘图、现有 `deep_gemm` Python API
- 可用：`subprocess` 调用 `nvidia-smi -lgc` 锁频
- 不可用：CUPTI PM Sampling（B200 上不完全可用）、`nvmlDeviceGetSamples(PROCESSOR_CLK)`（B200 不支持）

## Feasibility Hints and Suggestions

> **注意**：本节仅供参考，非强制要求。

### 概念方案

```
For each batch_size in [512, 1024, 8192, 32768]:
  1. 初始化 NVML，按 local_rank 获取设备句柄
  2. 预热 kernel（5 次）
  3. 实验 A（N 次迭代）：
     a. 读取 violation_status_before = nvmlDeviceGetViolationStatus(POWER)
     b. 启动 kernel（可选 profiler_buffer）
     c. cudaSynchronize
     d. 读取 violation_status_after
     e. 记录：wall_time, violation_delta, avg_sm_mhz
  4. 实验 B：
     a. 自由运行 N 次 → 记录 t_free, f_free
     b. subprocess: nvidia-smi -lgc 1965,1965
     c. 锁频运行 N 次 → 记录 t_locked, f_locked
     d. subprocess: nvidia-smi -rgc（重置）
     e. 计算敏感度指标
  5. 实验 C/D（后台线程）：
     a. 启动轮询线程（~1ms 间隔轮询 clock/throttle/power）
     b. 启动 kernel
     c. cudaSynchronize
     d. 停止轮询线程
     e. 收集带时间戳的样本
  6. 写入 CSV 输出
  7. 可选：生成 matplotlib 图表
```

### 相关参考
- `tests/test_mega_moe.py` — kernel 调用设置、profiler_buffer 布局、分布式初始化
- `deep_gemm/mega/__init__.py` — `fp8_fp4_mega_moe()` Python API，含 `enable_pull`/`enable_combine` 开关
- `deep_gemm/testing/bench.py` — `bench_kineto()` 计时参考
- `tests/test_mega_moe_profile.py` — 现有 `nvidia-smi` SM 时钟采集模式

### 已验证的 NVML API 可用性（B200, driver 590.48.01）

| API | 状态 | 备注 |
|-----|------|------|
| `nvmlDeviceGetClockInfo(SM)` | OK | 1-50 us 延迟，缓存读 |
| `nvmlDeviceGetViolationStatus(POWER)` | OK | 累计计数器 |
| `nvmlDeviceGetCurrentClocksThrottleReasons` | OK | 9 个 reason bit |
| `nvmlDeviceGetPowerUsage` | OK | mW 精度 |
| `nvmlDeviceGetSamples(TOTAL_POWER)` | OK | ~20ms 硬件采样 |
| `nvmlDeviceGetSamples(PROCESSOR_CLK)` | **不支持** | B200 限制 |
| `nvidia-smi -lgc` | OK | 锁频 |
| `nvmlDeviceGetSupportedGraphicsClocks` | OK | 247 档位, 120-1965 MHz |

### 已知限制
- NVML 是驱动缓存读，更新周期 ~10-100ms；对短 kernel（<3ms），violation 增量和前后快照比轮询更可靠
- `PROCESSOR_CLK` 硬件采样缓冲区在 B200 上不可用
- kernel 内 `clock64()` profiler 仍是 per-SM 频率的最精确手段（ns 精度）；NVML 提供互补的 host 侧数据

## Dependencies and Sequence

### 里程碑

1. **NVML 基础设施**：pynvml 封装、设备句柄管理、轮询线程
   - 阶段 A：NVML 初始化和前后快照辅助函数
   - 阶段 B：后台轮询线程，带时间戳的样本收集

2. **实验实现**：基于 NVML 基础设施的四个实验
   - 实验 A（violation 散点图）依赖里程碑 1 阶段 A
   - 实验 B（锁频对照）依赖里程碑 1 阶段 A + subprocess 时钟控制
   - 实验 C/D（轮询时间序列）依赖里程碑 1 阶段 B

3. **输出与可视化**：CSV 导出和可选图表
   - 依赖里程碑 1-2

4. **多 rank 集成**：按 rank 输出、分布式 kernel 设置
   - 依赖以上全部；复用 test_mega_moe.py 模式

## Task Breakdown

| Task ID | 描述 | 目标 AC | 标签 | 依赖 |
|---------|------|---------|------|------|
| task1 | 创建脚本骨架：argparse、NVML 初始化、分布式设置 | AC-7, AC-8 | `coding` | - |
| task2 | 实现前后 violation 快照（实验 A 核心） | AC-1 | `coding` | task1 |
| task3 | 实现后台轮询线程：clock/throttle/power | AC-2, AC-4 | `coding` | task1 |
| task4 | 实现硬件功耗采样获取（实验 D） | AC-3 | `coding` | task1 |
| task5 | 实现锁频对照（实验 B） | AC-5 | `coding` | task2 |
| task6 | 实现所有实验的 CSV 输出 | AC-6 | `coding` | task2, task3, task4 |
| task7 | 集成 kernel 内 profiler 交叉验证 | AC-8 | `coding` | task2 |
| task8 | 端到端测试：在 B512 和 B32k 上运行全部实验 | AC-1 至 AC-8 | `coding` | task6, task7 |

## Claude-Codex Deliberation

### 共识
- Codex 在计划生成期间不可用（DeepSeek 后端 API 与 Humanize 插件不兼容）。所有分析由 Claude 单独完成。

### 已解决的分歧
- 无（单方审查）

### 收敛状态
- 最终状态：`partially_converged`（仅 Claude，无 Codex 交叉审查）

## Pending User Decisions

- DEC-1: 实验 C/D 的轮询方式
  - Claude 立场：后台线程轮询 clock/throttle + 硬件功耗采样缓冲区
  - Codex 立场：N/A（不可用）
  - 权衡摘要：线程增加复杂度，但提供仅靠快照无法获得的 throttle reason 时间序列
  - 决策状态：`后台线程 + 硬件采样`（用户已确认）

- DEC-2: 实验 A 的迭代次数
  - Claude 立场：可配置，大约 100-500 范围
  - Codex 立场：N/A（不可用）
  - 权衡摘要：更多迭代 = 更好的统计，但运行时间更长
  - 决策状态：`约 100-500，可配置`（用户已确认）

## Implementation Notes

### 代码风格要求
- 实现代码和注释中不得包含计划特定术语，如 "AC-"、"Milestone"、"Step"、"Phase" 等工作流标记
- 这些术语仅用于计划文档，不应出现在代码库中
- 在代码中使用描述性、领域相关的命名

### 依赖
- `pynvml`（pip install nvidia-ml-py）— 必需
- `matplotlib` — 可选，用于绘图
- `torch`、`torch.distributed` — 现有依赖
- `deep_gemm` — 现有依赖

--- Original Design Draft Start ---

# Draft: 用 NVML API 精细量化降频情况及其原因

## 背景

DeepGEMM mega_moe kernel 在 B200 上运行时，NVLink 通信流量引发 DVFS 降频，导致 SM 频率从 1965 MHz 最大 boost 降至 ~1350 MHz（大 batch 时）。之前的实验（047/048）通过 kernel 内 `clock64() / globaltimer` profiler 测得了 per-SM 平均频率，但这只能测到时间区间内的平均值，无法观测频率随时间变化的动态行为。

## 目标

利用 NVML API 在 kernel 运行期间从 host 侧监控 GPU 状态，补充 kernel 内 profiler 无法覆盖的维度：

1. **Power violation 累计时间**：通过 `nvmlDeviceGetViolationStatus(POWER)` 的增量，量化每次 kernel 运行期间有多少时间受到功耗墙限制
2. **Throttle reason 分类**：通过 `nvmlDeviceGetCurrentClocksThrottleReasons` 区分 power cap、thermal、HW slowdown 等不同降频原因
3. **实时功耗曲线**：通过 `nvmlDeviceGetPowerUsage` 轮询，画出 kernel 运行期间的功耗时间序列
4. **SM clock 轮询**（辅助）：`nvmlDeviceGetClockInfo(SM)` 虽然分辨率有限（驱动缓存更新周期 ~10-100ms），但对 B32k 等长 kernel（~3ms）仍可提供粗粒度参考

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
