#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>

#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

namespace deep_gemm::sm100 {

template <uint32_t BLOCK_MN, uint32_t BLOCK_K>
__device__ __forceinline__
cute::UMMA::SmemDescriptor make_mxfp4_umma_desc(void* base_smem_ptr) {
    using layout_atom_t = cute::UMMA::Layout_K_SW128_Atom<cutlass::float_e2m1_t>;
    auto smem_layout = cute::tile_to_shape(layout_atom_t{}, cute::Shape<cute::Int<BLOCK_MN>, cute::Int<BLOCK_K>>{});
    auto smem_tensor = cute::make_tensor(cute::make_smem_ptr<cutlass::float_e2m1_t>(base_smem_ptr), smem_layout);
    return cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(smem_tensor);
}

template <typename tmem_tensor_t>
__device__ __forceinline__ uint32_t get_tmem_addr(const tmem_tensor_t& tensor) {
    return cute::raw_pointer_cast(tensor.data());
}

__device__ __forceinline__ uint32_t advance_mxfp4_umma_desc_lo(const uint32_t& base,
                                                               const uint32_t& offset,
                                                               const uint32_t& k_idx) {
    return base + ((offset + k_idx) >> 5u);
}

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode,
          bool kIs3DTMA = false>
__device__ __forceinline__ void
mxfp4_tma_copy(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
               uint8_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
               const uint32_t& num_tma_multicast = 1, const uint32_t& batch_idx = 0) {
    constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, uint8_t>() * 2;
    DG_STATIC_ASSERT(BLOCK_INNER % BLOCK_INNER_ATOM == 0, "Invalid MXFP4 TMA tile");
    DG_DEVICE_ASSERT(num_tma_multicast == 1);

    if constexpr (not kIs3DTMA) {
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
            cute::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                         static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                         smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                         inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
        }
    } else {
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
            cute::SM90_TMA_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                         static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                         smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                         inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
        }
    }
}

struct SM100_MMA_MXF4_SS {
    __device__ static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
            "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

} // namespace deep_gemm::sm100

namespace deep_gemm {

using namespace deep_gemm::sm100;

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t kGranKA, uint32_t kGranKB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t,
          typename epilogue_type_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_mxfp4_gemm_1d1d_impl(int* grouped_layout,
                           uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator1Sm;
    using ab_storage_t = uint8_t;
    using mma_dtype_t = cutlass::float_e2m1_t;
    using sf_dtype_t = cutlass::float_ue8m0_t;

    DG_STATIC_ASSERT(not kWithAccumulation, "MXFP4 kernel does not support accumulation");
    DG_STATIC_ASSERT(cute::is_same_v<a_dtype_t, ab_storage_t> and cute::is_same_v<b_dtype_t, ab_storage_t>,
                     "MXFP4 kernel expects uint8 byte storage for packed FP4 operands");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "MXFP4 kernel expects BF16 output");
    DG_STATIC_ASSERT(kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K,
                     "MXFP4 phase 1 only supports K-major operands");
    DG_STATIC_ASSERT(kGranKA == 32 and kGranKB == 32, "MXFP4 phase 1 requires granularity 32 for both operands");
    DG_STATIC_ASSERT(BLOCK_M == 128, "MXFP4 grouped-contiguous phase 1 requires BLOCK_M == 128");
    DG_STATIC_ASSERT(BLOCK_N % 16 == 0 and 16 <= BLOCK_N and BLOCK_N <= 128,
                     "MXFP4 grouped-contiguous phase 1 requires BLOCK_N to be a multiple of 16 in [16, 128]");
    DG_STATIC_ASSERT(BLOCK_K == 256, "MXFP4 grouped-contiguous phase 1 requires BLOCK_K == 256");
    DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "MXFP4 phase 1 only supports grouped contiguous GEMM");
    DG_STATIC_ASSERT(kNumMulticast == 1 and not kIsMulticastOnA, "MXFP4 phase 1 does not support multicast / 2SM MMA");
    DG_STATIC_ASSERT(kSwizzleAMode == 128 and kSwizzleBMode == 128,
                     "MXFP4 phase 1 expects canonical 128B swizzled K-major operand layouts");

    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 1;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t kNumSFAStagesPerLoad = 1;
    constexpr uint32_t kNumSFBStagesPerLoad = 1;
    DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const uint32_t shape_sfa_k = ceil_div(shape_k, kGranKA * 4);
    const uint32_t shape_sfb_k = ceil_div(shape_k, kGranKB * 4);

    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M, "Only support tensor memory layout A/D");
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SF_PACKED_K_PER_ROW = BLOCK_K / (kGranKA * 4);
    constexpr uint32_t SF_BLOCK_M = BLOCK_M;
    constexpr uint32_t SF_BLOCK_N = constexpr_align(BLOCK_N, 128u);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * SF_PACKED_K_PER_ROW * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * SF_PACKED_K_PER_ROW * sizeof(uint32_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype_t);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for UMMA");

    constexpr uint32_t kNumKBlocksPerTile = BLOCK_K / 64;
    constexpr uint32_t kNumSFATmemColsPerKBlock = 14;
    constexpr uint32_t kNumSFBTmemColsPerKBlock = (constexpr_align(BLOCK_N, 128u) == 128 ? 14 : 30);
    constexpr uint32_t kNumSFATmemCols = kNumKBlocksPerTile * kNumSFATmemColsPerKBlock;
    constexpr uint32_t kNumSFBTmemCols = kNumKBlocksPerTile * kNumSFBTmemColsPerKBlock;
    constexpr uint32_t kNumEpilogueStages = (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;

    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer +
        SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
        kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    auto full_barriers              = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers      = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });

    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            with_sf_full_barriers[i]->init(32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumUMMAStoreThreads);
        }

        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, 1, false, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx == 0 and cute::elect_one_sync()) {
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto& num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                uint32_t m_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<true, IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), IndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), IndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);

                constexpr bool kIsBatchedMM = false;
                constexpr uint32_t batch_idx = 0;
                mxfp4_tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, kIsBatchedMM>(
                    &tensor_map_a, full_barriers[stage_idx], reinterpret_cast<uint8_t*>(smem_a[stage_idx]), k_a_idx, m_idx, 1, batch_idx);
                mxfp4_tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, kIsBatchedMM>(
                    &tensor_map_b, full_barriers[stage_idx], reinterpret_cast<uint8_t*>(smem_b[stage_idx]), k_b_idx, n_idx, 1, batch_idx);
                auto num_arrival_bytes = SMEM_A_SIZE_PER_STAGE / 2 + SMEM_B_SIZE_PER_STAGE / 2;

                const auto sfa_k_idx = scheduler.template get_global_idx<false, IndexType::SF_K>(shape_sfa_k, SF_PACKED_K_PER_ROW, k_block_idx);
                const auto sfb_k_idx = scheduler.template get_global_idx<true, IndexType::SF_K>(shape_sfb_k, SF_PACKED_K_PER_ROW, k_block_idx, m_block_idx);
                #pragma unroll
                for (uint32_t i = 0; i < SF_PACKED_K_PER_ROW; ++ i) {
                    tma_copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx] + i * SF_BLOCK_M,
                                            m_block_idx * BLOCK_M, sfa_k_idx + i);
                    num_arrival_bytes += BLOCK_M * sizeof(uint32_t);
                    tma_copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx] + i * SF_BLOCK_N,
                                            n_block_idx * BLOCK_N, sfb_k_idx + i);
                    num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
                }

                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        constexpr uint32_t UMMA_M = LAYOUT_AD_M;
        constexpr uint32_t UMMA_N = BLOCK_N;
        constexpr uint32_t UMMA_K = 64;
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<mma_dtype_t, mma_dtype_t, float, sf_dtype_t,
                                                                   UMMA_M, UMMA_N, kMajorA, kMajorB>();
        using tiled_mma_t = decltype(cute::make_tiled_mma(
            cute::SM100_MMA_MXF4_SS<mma_dtype_t, mma_dtype_t, float, sf_dtype_t, BLOCK_M, BLOCK_N, 32, kMajorA, kMajorB>{}));
        using sm1xx_blk_scaled_config_t = cutlass::detail::Sm1xxBlockScaledConfig<32>;
        using smem_layout_atom_sfa_t = decltype(sm1xx_blk_scaled_config_t::deduce_smem_layoutSFA(
            tiled_mma_t{}, cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>>{}));
        using smem_layout_atom_sfb_t = decltype(sm1xx_blk_scaled_config_t::deduce_smem_layoutSFB(
            tiled_mma_t{}, cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>>{}));
        using smem_layout_sfa_t = decltype(cute::make_layout(
            cute::append(cute::shape(smem_layout_atom_sfa_t{}), cute::Int<kNumStages>{}),
            cute::append(cute::stride(smem_layout_atom_sfa_t{}), cute::size(cute::filter_zeros(smem_layout_atom_sfa_t{})))));
        using smem_layout_sfb_t = decltype(cute::make_layout(
            cute::append(cute::shape(smem_layout_atom_sfb_t{}), cute::Int<kNumStages>{}),
            cute::append(cute::stride(smem_layout_atom_sfb_t{}), cute::size(cute::filter_zeros(smem_layout_atom_sfb_t{})))));
        using utccp_t = cute::SM100_UTCCP_4x32dp128bit_1cta;

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = make_mxfp4_umma_desc<LOAD_BLOCK_M, BLOCK_K>(smem_a[0]);
        auto b_desc = make_mxfp4_umma_desc<LOAD_BLOCK_N, BLOCK_K>(smem_b[0]);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        auto tCtSFA = cute::make_tensor<typename tiled_mma_t::FrgTypeSFA>(cute::shape(smem_layout_atom_sfa_t{}));
        auto tCtSFB = cute::make_tensor<typename tiled_mma_t::FrgTypeSFB>(cute::shape(smem_layout_atom_sfb_t{}));
        tCtSFA.data() = cute::make_tmem_ptr<sf_dtype_t>(kTmemStartColOfSFA);
        tCtSFB.data() = cute::make_tmem_ptr<sf_dtype_t>(kTmemStartColOfSFB);
        auto tCtSFA_compact = cute::make_tensor(tCtSFA.data(), cute::filter_zeros(tCtSFA.layout()));
        auto tCtSFB_compact = cute::make_tensor(tCtSFB.data(), cute::filter_zeros(tCtSFB.layout()));
        auto tiled_copy_s2t_SFA = cute::make_utccp_copy(utccp_t{}, tCtSFA_compact);
        auto tiled_copy_s2t_SFB = cute::make_utccp_copy(utccp_t{}, tCtSFB_compact);
        auto thr_copy_s2t_SFA = tiled_copy_s2t_SFA.get_slice(0);
        auto thr_copy_s2t_SFB = tiled_copy_s2t_SFB.get_slice(0);
        auto thr_tCtSFA_compact_s2t = thr_copy_s2t_SFA.partition_D(tCtSFA_compact);
        auto thr_tCtSFB_compact_s2t = thr_copy_s2t_SFB.partition_D(tCtSFB_compact);
        auto tCsSFA_all = cute::make_tensor(cute::make_smem_ptr<sf_dtype_t>(smem_sfa[0]), smem_layout_sfa_t{});
        auto tCsSFB_all = cute::make_tensor(cute::make_smem_ptr<sf_dtype_t>(smem_sfb[0]), smem_layout_sfb_t{});
        auto tCsSFA_compact_all = cute::make_tensor(tCsSFA_all.data(), cute::filter_zeros(tCsSFA_all.layout()));
        auto tCsSFB_compact_all = cute::make_tensor(tCsSFB_all.data(), cute::filter_zeros(tCsSFB_all.layout()));
        auto thr_tCsSFA_compact_s2t_ = thr_copy_s2t_SFA.partition_S(tCsSFA_compact_all);
        auto thr_tCsSFB_compact_s2t_ = thr_copy_s2t_SFB.partition_S(tCsSFB_compact_all);
        auto thr_tCsSFA_compact_s2t = cute::get_utccp_smem_desc_tensor<utccp_t>(thr_tCsSFA_compact_s2t_);
        auto thr_tCsSFB_compact_s2t = cute::get_utccp_smem_desc_tensor<utccp_t>(thr_tCsSFB_compact_s2t_);
        auto tmem_sfa_addr = PatternVisitor([=](const uint32_t& k_block) {
            return get_tmem_addr(tCtSFA(cute::_, cute::_, k_block));
        });
        auto tmem_sfb_addr = PatternVisitor([=](const uint32_t& k_block) {
            return get_tmem_addr(tCtSFB(cute::_, cute::_, k_block));
        });

        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            tcgen05_after_thread_sync();

            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (do_tmem_full_arrive)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
            };

            const auto& num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                with_sf_full_barriers[stage_idx]->wait(phase);
                tcgen05_after_thread_sync();

                if (cute::elect_one_sync()) {
                    auto tCsSFA = tCsSFA_all(cute::_, cute::_, cute::_, stage_idx);
                    auto tCsSFB = tCsSFB_all(cute::_, cute::_, cute::_, stage_idx);
                    auto tCsSFA_compact = cute::make_tensor(tCsSFA.data(), cute::filter_zeros(tCsSFA.layout()));
                    auto tCsSFB_compact = cute::make_tensor(tCsSFB.data(), cute::filter_zeros(tCsSFB.layout()));
                    auto thr_tCsSFA_compact_s2t = cute::get_utccp_smem_desc_tensor<utccp_t>(thr_copy_s2t_SFA.partition_S(tCsSFA_compact));
                    auto thr_tCsSFB_compact_s2t = cute::get_utccp_smem_desc_tensor<utccp_t>(thr_copy_s2t_SFB.partition_S(tCsSFB_compact));
                    cute::copy(tiled_copy_s2t_SFA, thr_tCsSFA_compact_s2t, thr_tCtSFA_compact_s2t);
                    cute::copy(tiled_copy_s2t_SFB, thr_tCsSFB_compact_s2t, thr_tCtSFB_compact_s2t);
                }
                __syncwarp();

                const auto& a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc_block_scaled(instr_desc,
                                                                                                          tmem_sfa_addr[k],
                                                                                                          tmem_sfb_addr[k]);

                        b_desc.lo = advance_mxfp4_umma_desc_lo(b_desc_base_lo, 0, k * UMMA_K);
                        #pragma unroll
                        for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                            DG_STATIC_ASSERT((WAVE_BLOCK_M * BLOCK_K) % 128 == 0, "Invalid swizzling offset");
                            a_desc.lo = advance_mxfp4_umma_desc_lo(a_desc_base_lo, w * WAVE_BLOCK_M * BLOCK_K, k * UMMA_K);
                            SM100_MMA_MXF4_SS::fma(a_desc, b_desc,
                                                   accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                                   k_block_idx > 0 or k > 0,
                                                   runtime_instr_desc,
                                                   tmem_sfa_addr[k],
                                                   tmem_sfb_addr[k]);
                        }
                    }
                }

                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }
    } else if (warp_idx == 2) {
        auto utccp_required_smem_warp_transpose = [&](uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto& num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);
                cutlass::arch::fence_view_async_shared();

                #pragma unroll
                for (uint32_t i = 0; i < SF_PACKED_K_PER_ROW; ++ i) {
                    utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                    utccp_required_smem_warp_transpose(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                }
                __syncwarp();

                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        uint32_t tma_stage_idx = 0;
        auto advance_store_pipeline = [&]() {
            tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            tcgen05_after_thread_sync();

            DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Epilogue threads not enough");
            DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                #pragma unroll
                for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
                    if (epilogue_warp_idx == 0)
                        cute::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    const auto m_idx = scheduler.template get_global_idx<false, IndexType::MN>(shape_m, BLOCK_M, m_block_idx) + w * WAVE_BLOCK_M;
                    const auto n_idx = epilogue_type_t::apply_index_n<STORE_BLOCK_N>(n_block_idx * BLOCK_N + s * STORE_BLOCK_N);

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                        auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

                        constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                        col ^= row % (kSwizzleCDMode / 16);

                        uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N +
                                             w * BLOCK_N +
                                             s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                        auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                        epilogue_warp_idx * 32 * kSwizzleCDMode +
                                        row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                        uint32_t values[kNumElemsPerBankGroup];
                        DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                        cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                            values[0], values[1], values[2], values[3],
                            values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();
                        st_shared(smem_ptr,
                                  cast_into_bf16_and_pack(values[0], values[1]),
                                  cast_into_bf16_and_pack(values[2], values[3]),
                                  cast_into_bf16_and_pack(values[4], values[5]),
                                  cast_into_bf16_and_pack(values[6], values[7]));
                    }

                    if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                        tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        using cute_tma_t = cute::SM90_TMA_STORE_2D;
                        cute_tma_t::copy(&tensor_map_cd, smem_cd[tma_stage_idx], n_idx, m_idx);
                        cute::tma_store_arrive();
                    }
                }
            }
        }

        if (epilogue_warp_idx == kNumUMMAStoreThreads / 32 - 1)
            Allocator().free(0, kNumTmemCols);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
