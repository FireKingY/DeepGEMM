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
