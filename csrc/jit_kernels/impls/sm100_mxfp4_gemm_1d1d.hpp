#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

#include "runtime_utils.hpp"

namespace deep_gemm {

static CUtensorMapDataType aten_dtype_to_mxfp4_tensor_map_dtype(const at::ScalarType& dtype,
                                                                 const bool& allow_tf32) {
    if (dtype == kPackedFP4)
        return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    return aten_dtype_to_tensor_map_dtype(dtype, allow_tf32);
}

static CUtensorMap make_mxfp4_tma_2d_desc(const torch::Tensor& t,
                                          int gmem_inner_dim, int gmem_outer_dim,
                                          int smem_inner_dim, int smem_outer_dim,
                                          const int& gmem_outer_stride,
                                          const int& swizzle_mode, const int& swizzle_base = 0,
                                          const bool& allow_tf32 = false) {
    const auto& elem_size = static_cast<int>(t.element_size());
    if (swizzle_mode != 0) {
        smem_inner_dim = swizzle_mode / elem_size;
        if (t.scalar_type() == kPackedFP4)
            smem_inner_dim *= 2;
    }

    if (t.scalar_type() == kPackedFP4)
        DG_HOST_ASSERT(gmem_inner_dim % 128 == 0);

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * elem_size)};
    const cuuint32_t elem_strides[2] = {1, 1};
    DG_CUDA_DRIVER_CHECK(lazy_cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_mxfp4_tensor_map_dtype(t.scalar_type(), allow_tf32),
        2, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

static CUtensorMap make_mxfp4_tma_a_desc(const cute::UMMA::Major& major,
                                         const torch::Tensor& t,
                                         const int& shape_m, const int& shape_k,
                                         const int& block_m, const int& block_k,
                                         const int& outer_stride,
                                         const int& num_groups,
                                         const int& swizzle_mode, const int& swizzle_base = 0,
                                         const bool& allow_tf32 = false) {
    if (num_groups > 1)
        DG_HOST_ASSERT(major == cute::UMMA::Major::K);
    const auto& [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_m * num_groups);
    const auto& [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_m);
    return make_mxfp4_tma_2d_desc(t,
                                  gmem_inner_dim, gmem_outer_dim,
                                  smem_inner_dim, smem_outer_dim,
                                  outer_stride,
                                  swizzle_mode, swizzle_base,
                                  allow_tf32);
}

static CUtensorMap make_mxfp4_tma_b_desc(const cute::UMMA::Major& major,
                                         const torch::Tensor& t,
                                         const int& shape_n, const int& shape_k,
                                         const int& block_n, const int& block_k,
                                         const int& outer_stride,
                                         const int& num_groups,
                                         const int& swizzle_mode, const int& swizzle_base = 0,
                                         const bool& allow_tf32 = false) {
    const auto& [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_n);
    const auto& [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_n);
    return make_mxfp4_tma_2d_desc(t,
                                  gmem_inner_dim, gmem_outer_dim * num_groups,
                                  smem_inner_dim, smem_outer_dim,
                                  outer_stride,
                                  swizzle_mode, swizzle_base,
                                  allow_tf32);
}

class SM100MXFP4Gemm1D1DRuntime final: public LaunchRuntime<SM100MXFP4Gemm1D1DRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        int gran_k_a, gran_k_b;
        const std::string& compiled_dims;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_mxfp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mxfp4_gemm_1d1d_impl<
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}, {}, {},
        {}
    >);
}};
)",
        to_string(args.gemm_config.major_a), to_string(args.gemm_config.major_b),
        args.gran_k_a, args.gran_k_b,
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
        args.num_groups,
        args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode, args.gemm_config.smem_config.swizzle_cd_mode,
        args.gemm_config.num_stages,
        args.gemm_config.thread_config.num_non_epilogue_threads, args.gemm_config.thread_config.num_epilogue_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation,
        "uint8_t", "uint8_t", to_string(args.gemm_config.cd_dtype),
        get_default_epilogue_type(std::nullopt));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.grouped_layout, args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static void sm100_m_grouped_mxfp4_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                        const torch::Tensor& b, const torch::Tensor& sfb,
                                                        const torch::Tensor& d,
                                                        const torch::Tensor& grouped_layout,
                                                        const int& num_groups, const int& m, const int& n, const int& k,
                                                        const int& gran_k_a, const int& gran_k_b,
                                                        const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                        const std::string& compiled_dims) {
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::Kernel1D1D,
        m, n, k, num_groups, major_a, major_b,
        a.scalar_type(), b.scalar_type(),
        d.scalar_type(), false,
        device_runtime->get_num_sms(), std::optional<MmaKind>(MmaKind::MXFP4));

    const auto& tensor_map_a = make_mxfp4_tma_a_desc(major_a, a, m, k,
                                                     SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                                     config.block_k,
                                                     static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                                     config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_mxfp4_tma_b_desc(major_b, b, n, k,
                                                     SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                                     config.block_k,
                                                     static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                                     config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, gran_k_a, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, gran_k_b, num_groups, 0);

    const SM100MXFP4Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = num_groups,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100MXFP4Gemm1D1DRuntime::generate_impl(args);
    const auto& runtime = compiler->build("sm100_m_grouped_mxfp4_gemm_contiguous_1d1d", code);
    SM100MXFP4Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
