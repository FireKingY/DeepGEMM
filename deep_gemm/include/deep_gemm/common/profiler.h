#pragma once

#include <cstdint>

#define DG_PROFILER_STAGE_TAGS(prefix) \
  prefix##Stage0,                      \
  prefix##Stage1,                      \
  prefix##Stage2,                      \
  prefix##Stage3,                      \
  prefix##Stage4,                      \
  prefix##Stage5,                      \
  prefix##Stage6,                      \
  prefix##Stage7,                      \
  prefix##Stage8,                      \
  prefix##Stage9,                      \
  prefix##Stage10,                     \
  prefix##Stage11,                     \
  prefix##Stage12,                     \
  prefix##Stage13,                     \
  prefix##Stage14,                     \
  prefix##Stage15,                     \
  prefix##Stage16,                     \
  prefix##Stage17,                     \
  prefix##Stage18,                     \
  prefix##Stage19,                     \
  prefix##Stage20,                     \
  prefix##Stage21,                     \
  prefix##Stage22,                     \
  prefix##Stage23,                     \
  prefix##Stage24,                     \
  prefix##Stage25,                     \
  prefix##Stage26,                     \
  prefix##Stage27,                     \
  prefix##Stage28,                     \
  prefix##Stage29,                     \
  prefix##Stage30,                     \
  prefix##Stage31

enum ProfilerTag : int64_t {
  Init = 0,
  LoadWaitEmpty,
  LoadTmaIssue,
  DG_PROFILER_STAGE_TAGS(TmaToSfReady),
  SfWaitFull,
  SfTranspose,
  MmaWaitTmemEmpty,
  MmaWaitReady,
  UtccpScale,
  Umma,
  DG_PROFILER_STAGE_TAGS(UmmaToEmptyReady),
  EpilogueWaitFull,
  EpilogueStore,
};

#undef DG_PROFILER_STAGE_TAGS

__device__ inline int64_t globaltimer() {
    int64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

struct Profiler {
    int64_t* row_ptr_ = nullptr;
    int64_t max_entries_ = 0;
    int sm_id_ = 0;
    int count_ = 0;

    __device__ void init(int64_t* profiler_ptr, int64_t row_stride, int64_t slot_id, int64_t max_entries) {
        row_ptr_ = profiler_ptr + slot_id * row_stride;
        max_entries_ = max_entries;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id_));
        count_ = 0;
    }

    __device__ void start(ProfilerTag tag) {
        start_at(tag, globaltimer());
    }

    __device__ void start_at(ProfilerTag tag, int64_t start) {
        if (count_ >= max_entries_)
            return;
        row_ptr_[1 + count_ * 4 + 0] = sm_id_;
        row_ptr_[1 + count_ * 4 + 1] = static_cast<int64_t>(tag);
        row_ptr_[1 + count_ * 4 + 2] = start;
    }

    __device__ void stop() {
        stop_at(globaltimer());
    }

    __device__ void stop_at(int64_t end) {
        if (count_ >= max_entries_)
            return;
        row_ptr_[1 + count_ * 4 + 3] = end - row_ptr_[1 + count_ * 4 + 2];
        ++count_;
    }

    __device__ void flush() {
        row_ptr_[0] = count_;
    }

    __device__ void record(ProfilerTag tag, int64_t start, int64_t duration) {
        if (count_ >= max_entries_)
            return;
        row_ptr_[1 + count_ * 4 + 0] = sm_id_;
        row_ptr_[1 + count_ * 4 + 1] = static_cast<int64_t>(tag);
        row_ptr_[1 + count_ * 4 + 2] = start;
        row_ptr_[1 + count_ * 4 + 3] = duration;
        ++count_;
    }
};
