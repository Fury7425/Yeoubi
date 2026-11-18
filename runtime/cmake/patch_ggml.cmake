# SPDX-License-Identifier: Apache-2.0
# Patches a ggml source tree with the two kernels the Yeoubi runtime adds.
#
#   cmake -DGGML_SRC=<ggml checkout> -DPATCH_DIR=<runtime/patches> -P patch_ggml.cmake
#
# 1. A Vulkan kernel for GGML_OP_TOP_K_MASK. ggml defines the op and runs it on
#    CPU, CUDA, Metal, OpenCL and Hexagon; Vulkan had none, so TopK there
#    meant sorting every row.
# 2. A new op, GGML_OP_ACT_QUANT: Yeoubi's activation fake-quant as one
#    kernel per quantizer (CPU, Vulkan, Metal), instead of a chain of 6-13
#    tiny ops. Small iGPU steps are launch-bound, and these chains were over
#    a third of all ops. Backends without it keep the op chain (the runtime
#    probes supports_op), so nothing gets slower anywhere.
#
# Code is inserted at single-line anchors with string operations, not
# `git apply`: Windows checkouts may have CRLF line endings, which break
# diffs but not single-line matches. Each part is idempotent (an already
# patched tree is left alone); a missing anchor (newer ggml) fails loudly.

if(NOT GGML_SRC OR NOT PATCH_DIR)
    message(FATAL_ERROR "patch_ggml.cmake: pass -DGGML_SRC and -DPATCH_DIR")
endif()
set(V "${GGML_SRC}/src/ggml-vulkan")
set(C "${GGML_SRC}/src/ggml-cpu")
set(M "${GGML_SRC}/src/ggml-metal")

# Insert `text` before the anchor (BEFORE; the anchor must start its line) or
# after the end of the anchor's line (AFTER). An optional SCOPE string is
# found first and the anchor searched only after it (for repeated anchors).
function(insert file mode anchor text)
    file(READ "${file}" s)
    set(start 0)
    if(ARGC GREATER 4)
        string(FIND "${s}" "${ARGV4}" start)
        if(start EQUAL -1)
            message(FATAL_ERROR "patch_ggml.cmake: scope not found in ${file}: ${ARGV4}")
        endif()
    endif()
    string(SUBSTRING "${s}" ${start} -1 after)
    string(FIND "${after}" "${anchor}" rel)
    if(rel EQUAL -1)
        message(FATAL_ERROR "patch_ggml.cmake: anchor not found in ${file}: ${anchor}")
    endif()
    math(EXPR pos "${start} + ${rel}")
    if(mode STREQUAL "AFTER")
        string(SUBSTRING "${s}" ${pos} -1 tail)
        string(FIND "${tail}" "\n" nl)
        math(EXPR pos "${pos} + ${nl} + 1")
    endif()
    string(SUBSTRING "${s}" 0 ${pos} head)
    string(SUBSTRING "${s}" ${pos} -1 rest)
    file(WRITE "${file}" "${head}${text}${rest}")
endfunction()

function(has file marker out)
    file(READ "${file}" s)
    string(FIND "${s}" "${marker}" p)
    if(p EQUAL -1)
        set(${out} FALSE PARENT_SCOPE)
    else()
        set(${out} TRUE PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# 1. Vulkan TOP_K_MASK
# ---------------------------------------------------------------------------
has("${V}/ggml-vulkan-types.h" "pipeline_top_k_mask_f32" done)
if(NOT done)
    file(COPY "${PATCH_DIR}/top_k_mask.comp" DESTINATION "${V}/vulkan-shaders")

    insert("${V}/vulkan-shaders/vulkan-shaders-gen.cpp" AFTER
        "    string_to_spv(\"argmax_f32\""
        "    string_to_spv(\"top_k_mask_f32\", \"top_k_mask.comp\", merge_maps(base_dict, {{\"A_TYPE\", \"float\"}, {\"D_TYPE\", \"float\"}}));\n")

    insert("${V}/ggml-vulkan-types.h" AFTER
        "    vk_pipeline pipeline_argmax_f32;"
        "    vk_pipeline pipeline_top_k_mask_f32;\n")

    insert("${V}/ggml-vulkan.cpp" AFTER
        "    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, \"argmax_f32\""
        "    ggml_vk_create_pipeline(device, device->pipeline_top_k_mask_f32, \"top_k_mask_f32\", top_k_mask_f32_len, top_k_mask_f32_data, \"main\", 3, sizeof(vk_op_push_constants), {1, 1, 1}, { 256 }, 1);\n")

    # One workgroup per row: wg_denoms {1,1,1}, rows on y.
    insert("${V}/ggml-vulkan.cpp" BEFORE
        "void ggml_vk_argmax(ggml_backend_vk_context * ctx"
"static void ggml_vk_top_k_mask(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    vk_pipeline pipeline = ctx->device->pipeline_top_k_mask_f32;
    const uint32_t nrows = (uint32_t) ggml_nrows(dst);
    vk_op_push_constants pc { (uint32_t) dst->ne[0], nrows, (float) src1->ne[0], 0.0f, 0.0f, 0.0f };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, src0), ggml_vk_tensor_subbuffer(ctx, src1), ggml_vk_tensor_subbuffer(ctx, dst) },
        pc, { 1, nrows, 1 });
}

")

    insert("${V}/ggml-vulkan.cpp" AFTER
        "        ggml_vk_argmax(ctx, compute_ctx, src0, node);"
"        break;
    case GGML_OP_TOP_K_MASK:
        ggml_vk_top_k_mask(ctx, compute_ctx, src0, src1, node);
")

    insert("${V}/ggml-vulkan.cpp" BEFORE
        "        case GGML_OP_ARGMAX:"
"        case GGML_OP_TOP_K_MASK:
            // The shader covers the plain case: one mask row, no score/blocks/extra.
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_I32 && op->src[2] == nullptr &&
                   op->src[3] == nullptr && op->src[4] == nullptr &&
                   ggml_is_contiguous(op->src[0]) && ggml_nrows(op->src[0]) == 1 &&
                   ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op) &&
                   ggml_nrows(op->src[1]) == ggml_nrows(op) &&
                   op->ne[0] <= 8192 && ggml_nrows(op) <= 65535;
")
    message(STATUS "Patched ggml: Vulkan kernel for GGML_OP_TOP_K_MASK")
endif()

# ---------------------------------------------------------------------------
# 2. GGML_OP_ACT_QUANT: core op, CPU, Vulkan, Metal
# ---------------------------------------------------------------------------
has("${GGML_SRC}/include/ggml.h" "GGML_OP_ACT_QUANT" done)
if(NOT done)
    # --- core -------------------------------------------------------------
    insert("${GGML_SRC}/include/ggml.h" BEFORE
        "        GGML_OP_COUNT,"
        "        GGML_OP_ACT_QUANT, // Yeoubi activation fake-quant (runtime/cmake/patch_ggml.cmake)\n\n")

    insert("${GGML_SRC}/include/ggml.h" BEFORE
        "    GGML_API struct ggml_tensor * ggml_roll("
"    // Yeoubi activation fake-quant, per row of a contiguous F32 tensor:
    //   mode 0: INT4, s = sqrt(7) / mean|x|;  mode 1: INT4, s = 7 / max|x|;  mode 2: INT8, s = 127 / max|x|
    // mask (optional, same shape, F32: 0 keep, -inf drop) zeroes dropped values first.
    // out = clamp(round_half_even(x*s), -L, L) / s, with s = (1 / max(stat, 1e-8)) * level
    GGML_API struct ggml_tensor * ggml_act_quant(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * mask,
            int                   mode);

")

    file(READ "${GGML_SRC}/src/ggml.c" s)
    string(REGEX MATCH "GGML_OP_COUNT == ([0-9]+)" _ "${s}")
    set(old ${CMAKE_MATCH_1})
    if(NOT old)
        message(FATAL_ERROR "patch_ggml.cmake: GGML_OP_COUNT static_assert not found")
    endif()
    math(EXPR new "${old} + 1")
    string(REPLACE "GGML_OP_COUNT == ${old}, \"GGML_OP_COUNT != ${old}\"" "GGML_OP_COUNT == ${new}, \"GGML_OP_COUNT != ${new}\"" s "${s}")
    file(WRITE "${GGML_SRC}/src/ggml.c" "${s}")

    insert("${GGML_SRC}/src/ggml.c" AFTER "    \"GLU\"," "\n    \"ACT_QUANT\",\n")
    insert("${GGML_SRC}/src/ggml.c" AFTER "    \"glu(x)\"," "\n    \"act_quant(x)\",\n")
    insert("${GGML_SRC}/src/ggml.c" BEFORE
        "// ggml_roll"
"// ggml_act_quant

struct ggml_tensor * ggml_act_quant(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * mask,
        int                   mode) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 && ggml_is_contiguous(a));
    GGML_ASSERT(mode >= 0 && mode <= 2);
    if (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F32 && ggml_is_contiguous(mask) && ggml_are_same_shape(a, mask));
    }

    struct ggml_tensor * result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(result, 0, mode);

    result->op     = GGML_OP_ACT_QUANT;
    result->src[0] = a;
    result->src[1] = mask;

    return result;
}

")

    # --- CPU --------------------------------------------------------------
    insert("${C}/ops.h" AFTER
        "void ggml_compute_forward_roll(const struct"
        "void ggml_compute_forward_act_quant(const struct ggml_compute_params * params, struct ggml_tensor * dst);\n")

    insert("${C}/ops.cpp" BEFORE
        "void ggml_compute_forward_roll("
"// ggml_compute_forward_act_quant (Yeoubi, runtime/cmake/patch_ggml.cmake)

void ggml_compute_forward_act_quant(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * mask = dst->src[1];
    const int     mode  = ggml_get_op_params_i32(dst, 0);
    const int64_t n     = src0->ne[0];
    const int64_t nr    = ggml_nrows(src0);
    const float   level = mode == 2 ? 127.0f : 7.0f;

    for (int64_t r = params->ith; r < nr; r += params->nth) {
        const float * x = (const float *) src0->data + r*n;
        float       * y = (float *) dst->data + r*n;
        if (mask) {  // dropped values -> 0 into y, then work on y in place
            const float * m = (const float *) mask->data + r*n;
            for (int64_t i = 0; i < n; i++) {
                y[i] = m[i] > -INFINITY ? x[i] : 0.0f;
            }
            x = y;
        }
        // Branch-free loops, so compilers can vectorize them.
        float acc = 0.0f;
        if (mode == 0) {
            for (int64_t i = 0; i < n; i++) {
                acc += std::fabs(x[i]);
            }
            acc /= (float) n;
        } else {
            for (int64_t i = 0; i < n; i++) {
                acc = std::max(acc, std::fabs(x[i]));
            }
        }
        const float stat = std::max(acc, 1e-8f);
        const float s    = (1.0f / stat) * (mode == 0 ? 2.6457513110645907f : level);
        for (int64_t i = 0; i < n; i++) {
            // nearbyint rounds half to even under the default mode, as torch.round does
            y[i] = std::min(std::max(std::nearbyint(x[i]*s), -level), level) / s;
        }
    }
}

")

    insert("${C}/ggml-cpu.c" AFTER
        "                ggml_compute_forward_roll(params, tensor);"
"            } break;
        case GGML_OP_ACT_QUANT:
            {
                ggml_compute_forward_act_quant(params, tensor);
")

    # Rows split across all threads (the COUNT_EQUAL group), not n_tasks = 1.
    insert("${C}/ggml-cpu.c" BEFORE
        "        case GGML_OP_COUNT_EQUAL:"
        "        case GGML_OP_ACT_QUANT:\n"
        "static int ggml_get_n_tasks(")

    # --- Vulkan -----------------------------------------------------------
    file(COPY "${PATCH_DIR}/act_quant.comp" DESTINATION "${V}/vulkan-shaders")

    insert("${V}/vulkan-shaders/vulkan-shaders-gen.cpp" AFTER
        "    string_to_spv(\"argmax_f32\""
        "    string_to_spv(\"act_quant_f32\", \"act_quant.comp\", merge_maps(base_dict, {{\"A_TYPE\", \"float\"}, {\"D_TYPE\", \"float\"}}));\n")

    insert("${V}/ggml-vulkan-types.h" AFTER
        "    vk_pipeline pipeline_argmax_f32;"
        "    vk_pipeline pipeline_act_quant_f32;\n")

    insert("${V}/ggml-vulkan.cpp" AFTER
        "    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, \"argmax_f32\""
        "    ggml_vk_create_pipeline(device, device->pipeline_act_quant_f32, \"act_quant_f32\", act_quant_f32_len, act_quant_f32_data, \"main\", 3, sizeof(vk_op_push_constants), {1, 1, 1}, { 256 }, 1);\n")

    insert("${V}/ggml-vulkan.cpp" BEFORE
        "void ggml_vk_argmax(ggml_backend_vk_context * ctx"
"static void ggml_vk_act_quant(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    vk_pipeline pipeline = ctx->device->pipeline_act_quant_f32;
    const uint32_t nrows = (uint32_t) ggml_nrows(dst);
    vk_op_push_constants pc { (uint32_t) dst->ne[0], nrows, (float) ggml_get_op_params_i32(dst, 0), src1 ? 1.0f : 0.0f, 0.0f, 0.0f };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, src0), ggml_vk_tensor_subbuffer(ctx, src1 ? src1 : src0), ggml_vk_tensor_subbuffer(ctx, dst) },
        pc, { 1, nrows, 1 });
}

")

    insert("${V}/ggml-vulkan.cpp" AFTER
        "        ggml_vk_argmax(ctx, compute_ctx, src0, node);"
"        break;
    case GGML_OP_ACT_QUANT:
        ggml_vk_act_quant(ctx, compute_ctx, src0, src1, node);
")

    insert("${V}/ggml-vulkan.cpp" BEFORE
        "        case GGML_OP_ARGMAX:"
"        case GGML_OP_ACT_QUANT:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op) && ggml_nrows(op) <= 65535 &&
                   (op->src[1] == nullptr || (op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1])));
")

    # --- Metal ------------------------------------------------------------
    file(READ "${PATCH_DIR}/act_quant.metal" kernel)
    file(APPEND "${M}/kernels/misc.metal" "${kernel}")

    # Declared in the header: the op dispatch switch comes before the definitions.
    insert("${M}/ggml-metal-ops.h" AFTER
        "int ggml_metal_op_argmax            (ggml_metal_op_t ctx, int idx);"
        "int ggml_metal_op_act_quant         (ggml_metal_op_t ctx, int idx);\n")

    insert("${M}/ggml-metal-ops.cpp" BEFORE
        "int ggml_metal_op_argmax(ggml_metal_op_t ctx, int idx) {"
"int ggml_metal_op_act_quant(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, \"kernel_act_quant_f32\");
    if (!pipeline.pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, \"kernel_act_quant_f32\", \"kernel_act_quant_f32\", nullptr);
    }

    const bool has_mask = op->src[1] != nullptr;
    ggml_metal_kargs_argmax args = {
        /*.ne00  = */ op->ne[0],
        /*.nb01  = */ op->nb[1],
        /*.npart = */ ggml_get_op_params_i32(op, 0) | (has_mask ? 16 : 0),
        /*.ne01  = */ ggml_nrows(op),
    };

    int nth = 32; // SIMD width
    while (nth < op->ne[0] && nth < 256) {
        nth *= 2;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(has_mask ? op->src[1] : op->src[0]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 3);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, 32*sizeof(float), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, ggml_nrows(op), 1, 1, nth, 1, 1);

    return 1;
}

")

    insert("${M}/ggml-metal-ops.cpp" AFTER
        "                n_fuse = ggml_metal_op_argmax(ctx, idx);"
"            } break;
        case GGML_OP_ACT_QUANT:
            {
                n_fuse = ggml_metal_op_act_quant(ctx, idx);
")

    insert("${M}/ggml-metal-device.m" BEFORE
        "        case GGML_OP_ARGMAX:"
"        case GGML_OP_ACT_QUANT:
            return has_simdgroup_reduction && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op) &&
                (op->src[1] == NULL || (op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1])));
")
    message(STATUS "Patched ggml: GGML_OP_ACT_QUANT (core, CPU, Vulkan, Metal)")
endif()
