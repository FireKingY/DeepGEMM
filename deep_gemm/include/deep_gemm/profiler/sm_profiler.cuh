// Vendored device-only mirror of ~/ssd/sm-profiler/cuda/sm_profiler.h.
// Binary-compatible buffer / event layout — the host-side library can parse
// & export traces produced by this header directly.
//
// Skips the host-side `extern "C"` declarations (NVRTC-friendly).
//
// Provides BOTH APIs:
//   - "full" events: 40B/event, globaltimer ns timestamps, sm_id per event.
//   - "light" events: 24B/event, clock64() cycles, single-init context cached
//     across calls (one buffer-header parse per warp instead of per call).

#pragma once

#include <cstdint>

#define SM_PROFILER_MAX_EVENT_TYPES 32

#define SM_PROFILER_EVENT_TYPE_RANGE 0u
#define SM_PROFILER_EVENT_TYPE_INSTANT 1u

#define SM_PROFILER_BUFFER_FORMAT_FULL 0u
#define SM_PROFILER_BUFFER_FORMAT_LIGHT 1u
#define SM_PROFILER_BUFFER_VERSION_V1 1u

#define SM_PROFILER_BUFFER_FLAG_FORMAT_SHIFT 0u
#define SM_PROFILER_BUFFER_FLAG_FORMAT_MASK 0xFFFFu
#define SM_PROFILER_BUFFER_FLAG_VERSION_SHIFT 16u
#define SM_PROFILER_BUFFER_FLAG_VERSION_MASK 0xFFFF0000u

struct SmProfilerActiveEventEntry {
    uint32_t active_event_id;
};

struct __attribute__((packed)) SmProfilerDeviceEvent {
    uint32_t event_id;
    uint32_t event_no;
    uint32_t block_id;
    uint32_t group_idx;
    uint32_t sm_id;
    uint32_t type;
    uint64_t st_timestamp_ns;
    uint64_t en_timestamp_ns;
};

struct __align__(8) SmProfilerLightEvent {
    uint64_t st_timestamp;
    uint64_t en_timestamp;
    uint32_t event_meta;
    uint32_t reserved;
};

struct sm_profiler_light_range_t {
    SmProfilerLightEvent* event;
};

__device__ __forceinline__ uint32_t sm_profiler_get_block_idx() {
    return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

__device__ __forceinline__ uint32_t sm_profiler_get_thread_idx_in_block() {
    return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ uint32_t sm_profiler_get_lane_id() {
    uint32_t lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__ __forceinline__ uint32_t sm_profiler_get_warp_id() {
    return sm_profiler_get_thread_idx_in_block() / 32;
}

__device__ __forceinline__ uint32_t sm_profiler_get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ __forceinline__ uint64_t sm_profiler_get_timestamp() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
    return ret;
}

__device__ __forceinline__ uint64_t sm_profiler_get_timestamp_cycles() {
    return clock64();
}

struct __SmProfilerBufferInfo {
    uint32_t num_blocks;
    uint32_t num_groups;
    uint32_t max_events_per_group;
    uint32_t enabled;
    uint32_t flags;
    uint32_t buffer_format;
    uint32_t buffer_version;
    uint32_t event_size_bytes;
    uint32_t* counters;
    char* active_table_base;
    char* event_data_base;
};

__device__ __forceinline__ __SmProfilerBufferInfo __sm_profiler_parse_buffer(uint64_t* buffer) {
    __SmProfilerBufferInfo info;

    uint64_t header0 = buffer[0];
    uint64_t header1 = buffer[1];
    uint64_t header2 = buffer[2];
    info.num_blocks = static_cast<uint32_t>(header0 >> 32);
    info.num_groups = static_cast<uint32_t>(header0 & 0xFFFFFFFFu);
    info.max_events_per_group = static_cast<uint32_t>(header1 >> 32);
    info.enabled = static_cast<uint32_t>(header1 & 0xFFFFFFFFu);
    info.flags = static_cast<uint32_t>(header2 & 0xFFFFFFFFu);
    info.buffer_format = (info.flags >> SM_PROFILER_BUFFER_FLAG_FORMAT_SHIFT) & SM_PROFILER_BUFFER_FLAG_FORMAT_MASK;
    info.buffer_version = (info.flags >> SM_PROFILER_BUFFER_FLAG_VERSION_SHIFT) & 0xFFFFu;
    info.event_size_bytes = info.buffer_format == SM_PROFILER_BUFFER_FORMAT_LIGHT ?
        static_cast<uint32_t>(sizeof(SmProfilerLightEvent)) :
        static_cast<uint32_t>(sizeof(SmProfilerDeviceEvent));

    info.counters = reinterpret_cast<uint32_t*>(buffer + 3);

    size_t counters_bytes = static_cast<size_t>(info.num_blocks) * info.num_groups * sizeof(uint32_t);
    uint32_t counters_size_uint64 = static_cast<uint32_t>((counters_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    uint32_t active_offset = 3 + counters_size_uint64;

    size_t active_bytes = info.buffer_format == SM_PROFILER_BUFFER_FORMAT_LIGHT ? 0 :
        static_cast<size_t>(info.num_blocks) * info.num_groups *
        SM_PROFILER_MAX_EVENT_TYPES * sizeof(SmProfilerActiveEventEntry);
    uint32_t active_size_uint64 = static_cast<uint32_t>((active_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    uint32_t event_offset = active_offset + active_size_uint64;

    info.active_table_base = reinterpret_cast<char*>(buffer + active_offset);
    info.event_data_base = reinterpret_cast<char*>(buffer + event_offset);
    return info;
}

__device__ __forceinline__ SmProfilerActiveEventEntry* __sm_profiler_get_active_entry(
    __SmProfilerBufferInfo& info, uint32_t block_idx, uint32_t group_idx, uint32_t event_no) {
    size_t block_stride = static_cast<size_t>(info.num_groups) *
        SM_PROFILER_MAX_EVENT_TYPES * sizeof(SmProfilerActiveEventEntry);
    size_t group_stride = SM_PROFILER_MAX_EVENT_TYPES * sizeof(SmProfilerActiveEventEntry);
    return reinterpret_cast<SmProfilerActiveEventEntry*>(
        info.active_table_base + block_idx * block_stride + group_idx * group_stride +
        event_no * sizeof(SmProfilerActiveEventEntry));
}

__device__ __forceinline__ SmProfilerDeviceEvent* __sm_profiler_get_event(
    __SmProfilerBufferInfo& info, uint32_t block_idx, uint32_t group_idx, uint32_t event_idx) {
    size_t group_offset = (static_cast<size_t>(block_idx) * info.num_groups + group_idx) * info.max_events_per_group;
    return reinterpret_cast<SmProfilerDeviceEvent*>(info.event_data_base + (group_offset + event_idx) * info.event_size_bytes);
}

__device__ __forceinline__ SmProfilerLightEvent* __sm_profiler_get_light_event(
    __SmProfilerBufferInfo& info, uint32_t block_idx, uint32_t group_idx, uint32_t event_idx) {
    size_t group_offset = (static_cast<size_t>(block_idx) * info.num_groups + group_idx) * info.max_events_per_group;
    return reinterpret_cast<SmProfilerLightEvent*>(info.event_data_base + (group_offset + event_idx) * info.event_size_bytes);
}

// ───────── Full API (per-call buffer parse, 40B/event, %globaltimer) ─────────

__device__ __forceinline__ void sm_profiler_event_start(
    uint64_t* buffer, uint32_t event_no, bool predicate)
{
    if (!predicate || buffer == nullptr) return;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return;

    __SmProfilerBufferInfo info = __sm_profiler_parse_buffer(buffer);
    if (!info.enabled || info.buffer_format != SM_PROFILER_BUFFER_FORMAT_FULL) return;

    uint32_t block_idx = sm_profiler_get_block_idx();
    uint32_t group_idx = sm_profiler_get_warp_id();
    if (block_idx >= info.num_blocks || group_idx >= info.num_groups) return;

    SmProfilerActiveEventEntry* entry = __sm_profiler_get_active_entry(info, block_idx, group_idx, event_no);
    if (entry->active_event_id != 0) return;

    uint32_t* counter = &info.counters[block_idx * info.num_groups + group_idx];
    uint32_t event_id = *counter;
    *counter = event_id + 1;
    if (event_id >= info.max_events_per_group) return;

    SmProfilerDeviceEvent* ev = __sm_profiler_get_event(info, block_idx, group_idx, event_id);
    ev->event_id = event_id;
    ev->event_no = event_no;
    ev->block_id = block_idx;
    ev->group_idx = group_idx;
    ev->sm_id = sm_profiler_get_smid();
    ev->type = SM_PROFILER_EVENT_TYPE_RANGE;
    asm volatile("" ::: "memory");
    ev->st_timestamp_ns = sm_profiler_get_timestamp();
    ev->en_timestamp_ns = 0;
    entry->active_event_id = event_id + 1;
}

__device__ __forceinline__ void sm_profiler_event_end(
    uint64_t* buffer, uint32_t event_no, bool predicate)
{
    if (!predicate || buffer == nullptr) return;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return;

    __SmProfilerBufferInfo info = __sm_profiler_parse_buffer(buffer);
    if (!info.enabled || info.buffer_format != SM_PROFILER_BUFFER_FORMAT_FULL) return;

    uint32_t block_idx = sm_profiler_get_block_idx();
    uint32_t group_idx = sm_profiler_get_warp_id();
    if (block_idx >= info.num_blocks || group_idx >= info.num_groups) return;

    SmProfilerActiveEventEntry* entry = __sm_profiler_get_active_entry(info, block_idx, group_idx, event_no);
    if (entry->active_event_id == 0) return;
    uint32_t event_id = entry->active_event_id - 1;

    SmProfilerDeviceEvent* ev = __sm_profiler_get_event(info, block_idx, group_idx, event_id);
    ev->en_timestamp_ns = sm_profiler_get_timestamp();
    entry->active_event_id = 0;
}

__device__ __forceinline__ void sm_profiler_event_instant(
    uint64_t* buffer, uint32_t event_no, bool predicate)
{
    if (!predicate || buffer == nullptr) return;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return;

    __SmProfilerBufferInfo info = __sm_profiler_parse_buffer(buffer);
    if (!info.enabled || info.buffer_format != SM_PROFILER_BUFFER_FORMAT_FULL) return;

    uint32_t block_idx = sm_profiler_get_block_idx();
    uint32_t group_idx = sm_profiler_get_warp_id();
    if (block_idx >= info.num_blocks || group_idx >= info.num_groups) return;

    uint32_t* counter = &info.counters[block_idx * info.num_groups + group_idx];
    uint32_t event_id = *counter;
    *counter = event_id + 1;
    if (event_id >= info.max_events_per_group) return;

    SmProfilerDeviceEvent* ev = __sm_profiler_get_event(info, block_idx, group_idx, event_id);
    ev->event_id = event_id;
    ev->event_no = event_no;
    ev->block_id = block_idx;
    ev->group_idx = group_idx;
    ev->sm_id = sm_profiler_get_smid();
    ev->type = SM_PROFILER_EVENT_TYPE_INSTANT;
    ev->st_timestamp_ns = sm_profiler_get_timestamp();
    ev->en_timestamp_ns = 0;
}

// ───────── Light API (single context init, 24B/event, clock64() cycles) ─────────

struct sm_profiler_light_context_t {
    uint32_t enabled;
    uint32_t max_events_per_group;
    uint32_t next_event_id;
    uint32_t* counter;
    SmProfilerLightEvent* events;
};

__device__ __forceinline__ uint32_t sm_profiler_light_make_event_meta(uint32_t event_no, uint32_t event_type) {
    return (event_type << 16) | (event_no & 0xFFFFu);
}

__device__ __forceinline__ uint32_t sm_profiler_light_event_no_from_meta(uint32_t event_meta) {
    return event_meta & 0xFFFFu;
}

__device__ __forceinline__ uint32_t sm_profiler_light_event_type_from_meta(uint32_t event_meta) {
    return (event_meta >> 16) & 0xFFFFu;
}

__device__ __forceinline__ sm_profiler_light_context_t sm_profiler_init_light_context(uint64_t* buffer) {
    sm_profiler_light_context_t ctx = {};
    if (buffer == nullptr) return ctx;

    __SmProfilerBufferInfo info = __sm_profiler_parse_buffer(buffer);
    if (!info.enabled || info.buffer_format != SM_PROFILER_BUFFER_FORMAT_LIGHT) return ctx;

    const uint32_t block_idx = sm_profiler_get_block_idx();
    const uint32_t group_idx = sm_profiler_get_warp_id();
    if (block_idx >= info.num_blocks || group_idx >= info.num_groups) return ctx;

    ctx.enabled = 1;
    ctx.max_events_per_group = info.max_events_per_group;
    ctx.counter = &info.counters[block_idx * info.num_groups + group_idx];
    ctx.next_event_id = *ctx.counter;
    ctx.events = __sm_profiler_get_light_event(info, block_idx, group_idx, 0);
    return ctx;
}

__device__ __forceinline__ void sm_profiler_flush_light_context(
    const sm_profiler_light_context_t& ctx)
{
    if (!ctx.enabled) return;
    *ctx.counter = ctx.next_event_id;
}

__device__ __forceinline__ sm_profiler_light_range_t sm_profiler_event_start_light(
    sm_profiler_light_context_t& ctx, uint32_t event_no, bool predicate)
{
    sm_profiler_light_range_t range = {nullptr};
    if (!predicate || !ctx.enabled) return range;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return range;

    uint32_t event_id = ctx.next_event_id;
    if (event_id >= ctx.max_events_per_group) return range;
    ctx.next_event_id = event_id + 1;

    SmProfilerLightEvent* ev = &ctx.events[event_id];
    ev->st_timestamp = sm_profiler_get_timestamp_cycles();
    ev->en_timestamp = 0;
    ev->event_meta = sm_profiler_light_make_event_meta(event_no, SM_PROFILER_EVENT_TYPE_RANGE);
    ev->reserved = 0;
    range.event = ev;
    return range;
}

__device__ __forceinline__ void sm_profiler_event_end_light(
    const sm_profiler_light_range_t& range, bool predicate)
{
    if (!predicate || range.event == nullptr) return;
    range.event->en_timestamp = sm_profiler_get_timestamp_cycles();
}

__device__ __forceinline__ void sm_profiler_event_instant_light(
    sm_profiler_light_context_t& ctx, uint32_t event_no, bool predicate)
{
    if (!predicate || !ctx.enabled) return;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return;

    uint32_t event_id = ctx.next_event_id;
    if (event_id >= ctx.max_events_per_group) return;
    ctx.next_event_id = event_id + 1;

    SmProfilerLightEvent* ev = &ctx.events[event_id];
    ev->st_timestamp = sm_profiler_get_timestamp_cycles();
    ev->en_timestamp = 0;
    ev->event_meta = sm_profiler_light_make_event_meta(event_no, SM_PROFILER_EVENT_TYPE_INSTANT);
    ev->reserved = 0;
}
