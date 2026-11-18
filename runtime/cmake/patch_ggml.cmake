# SPDX-License-Identifier: Apache-2.0
# Adds a Vulkan kernel for GGML_OP_TOP_K_MASK to a ggml source tree.
#
#   cmake -DGGML_SRC=<ggml checkout> -DPATCH_DIR=<runtime/patches> -P patch_ggml.cmake
#
# ggml defines TOP_K_MASK and runs it on CPU, CUDA, Metal, OpenCL and Hexagon,
# but Vulkan has no kernel. The runtime builds TopK sparsification from
# ggml_top_k + TOP_K_MASK where a device supports both; without a Vulkan
# kernel the Arc/Radeon/Adreno path had to sort every row instead.
#
# Code is inserted at single-line anchors with string operations, not
# `git apply`: Windows checkouts may have CRLF line endings, which break
# diffs but not single-line matches. Idempotent: an already patched tree is
# left alone. A missing anchor (newer ggml) fails loudly.

if(NOT GGML_SRC OR NOT PATCH_DIR)
    message(FATAL_ERROR "patch_ggml.cmake: pass -DGGML_SRC and -DPATCH_DIR")
endif()
set(V "${GGML_SRC}/src/ggml-vulkan")

# Insert `text` before the anchor (MODE BEFORE; anchor must start its line)
# or after the end of the anchor's line (MODE AFTER).
function(insert file mode anchor text)
    file(READ "${file}" s)
    string(FIND "${s}" "${anchor}" pos)
    if(pos EQUAL -1)
        message(FATAL_ERROR "patch_ggml.cmake: anchor not found in ${file}: ${anchor}")
    endif()
    if(mode STREQUAL "AFTER")
        string(SUBSTRING "${s}" ${pos} -1 tail)
        string(FIND "${tail}" "\n" nl)
        math(EXPR pos "${pos} + ${nl} + 1")
    endif()
    string(SUBSTRING "${s}" 0 ${pos} head)
    string(SUBSTRING "${s}" ${pos} -1 rest)
    file(WRITE "${file}" "${head}${text}${rest}")
endfunction()

file(READ "${V}/ggml-vulkan-types.h" types)
string(FIND "${types}" "pipeline_top_k_mask_f32" done)
if(NOT done EQUAL -1)
    message(STATUS "ggml already has the Vulkan TOP_K_MASK kernel")
    return()
endif()

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
