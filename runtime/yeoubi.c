// SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
// Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
// Source code is licensed under Apache 2.0. See LICENSE for details.
/*
 * Native Yeoubi runtime on ggml.
 *
 * One graph, every ggml backend: CUDA, HIP/ROCm, Vulkan, Metal, SYCL,
 * OpenVINO, Hexagon, OpenCL/Adreno, CPU. The backend is picked at run time
 * (--device, default: ggml_backend_init_best); ops a backend lacks fall back
 * to the CPU through ggml_backend_sched, so every build produces the same
 * numbers, only faster or slower.
 *
 * Input is a GGUF from `python export.py --format gguf` or the torch-free
 * `python -m yeoubi.gguf_export` (yeoubi/gguf_export.py). The forward pass
 * mirrors yeoubi/model.py + yeoubi/rdt.py at eval time, MoE included; the
 * samplers mirror yeoubi/sample.py: denoise() (full canvas) and
 * BlockDiffusionSampler (--block, quantized KV cache). Check parity with
 * scripts/check_runtime_parity.py before trusting a new build.
 */

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define MASK_ID   258
#define EOS_ID    257
#define VOCAB     258   // predictable IDs: bytes 0-255, PAD, EOS (byte_codec.py)
#define MAX_LOOPS 256
#define MAX_TOPK  16    // experts per token
#define N_CONST   64    // distinct scalar constants per graph
#define ZEROS_MAX 8192  // widest row the fast TopK path handles (the Vulkan kernel's limit)

typedef struct ggml_tensor ten;

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "yeoubi: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

enum { ROPE_NONE, ROPE_LINEAR, ROPE_YARN };
enum { ACT_A4, ACT_A8, ACT_NONE };          // --act: trained stage / A8 stage / off
enum { Q_INT4, Q_TOPK, Q_INT8 };            // a BitLinear's configured input quantizer
enum { KV_NONE, KV_READ, KV_COMMIT };       // full canvas / denoise a block / commit a block

// Dense BitFFN (up = fused [gate; up] rows), or BitMoEFFN when router is set
// (up/gate/down are then stacked [in, out, n_experts], sub [ffn_dim, n_experts]).
typedef struct { ten *up, *gate, *down, *sub, *router; } ffn_t;

typedef struct {
    ten *attn_norm, *ffn_norm;
    ten *wqkv, *wo, *q_norm, *k_norm, *attn_sub;  // wqkv: fused [q; k; v] rows
    ffn_t ffn;
    int tidx;  // this block's slice of the fused t_proj
} block_t;

typedef struct {
    // config
    int    n_heads, head_dim, hidden, patch, t_dim, loop_dim, max_loops, min_loops;
    int    int4_absmean, adaptive_loops, context_adaptive, context_threshold, rope_type;
    int    n_experts, top_k, kv_bits, kv_bos_bits;
    double topk_ratio, rope_theta, rope_factor, timestep_scale, state_std, capacity_factor;
    // runtime
    int act, fast_topk, fused_quant, fused_topk_max;  // fused_topk_max: widest row for fused TopK (0 = none)
    int flash_attn;
    int views_compact;  // backend ignores views (OpenVINO): make every split compact
    ggml_backend_t backend, cpu;
    ggml_backend_sched_t sched;
    struct ggml_context *wctx;
    ggml_backend_buffer_t wbuf;
    // weights (NULL = absent in this checkpoint)
    ten *embed, *head, *ne0_w, *ne0_b, *ne2_w, *ne2_b, *final_norm, *self_cond_proj;  // head: unembed, = embed when tied
    ten *subpos, *enc_norm, *enc_conv, *cmp_norm, *cmp, *exp, *dec_norm, *dec_conv;
    ten *dec_ffn_norm, *out_norm;
    ffn_t dec_ffn;
    block_t *pre, *rec, *post;   // pre = prelude + pre_recurrent (or blocks), post = post_recurrent + coda
    int n_pre, n_rec, n_post, is_rdt, has_moe;
    ten *inj_A, *inj_B, *lora_down, *lora_up, *act_w, *loop_proj, *loop_proj2, *loop_proj2_b, *loop_norm;
    ten *loop_norms[MAX_LOOPS];
    int n_loop_norms;
    float *alpha, *iter_gate, *loop_embed;  // tiny host copies used as graph constants
    int n_alpha, n_gate, n_loop_embed;
    ten *tproj_all;                          // every block's t_proj, stacked
    char (*tproj_names)[GGML_MAX_NAME];
    int n_tproj;
    // KV cache for the block sampler: one K and V per virtual layer, holding
    // the dequantized values of PyTorch's quantized KVCache.
    struct ggml_context *kvctx;
    ggml_backend_buffer_t kvbuf;
    ten **kv_k, **kv_v;
    int kv_layers, kv_cap, kv_len, kv_loops;  // kv_cap / kv_len in global (patch) positions
} model_t;

static ten *get(model_t *m, const char *name) { return ggml_get_tensor(m->wctx, name); }

static ten *need(model_t *m, const char *name) {
    ten *t = get(m, name);
    if (!t) die("missing tensor %s", name);
    return t;
}

static ten *getf(model_t *m, int required, const char *fmt, ...) {
    char name[GGML_MAX_NAME];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    return required ? need(m, name) : get(m, name);
}

static void load_ffn(model_t *m, ffn_t *f, const char *p) {
    f->router = getf(m, 0, "%s.router.weight", p);
    if (f->router) {
        f->up   = getf(m, 1, "%s.experts.up_proj.weight", p);
        f->gate = getf(m, 1, "%s.experts.gate_proj.weight", p);
        f->down = getf(m, 1, "%s.experts.down_proj.weight", p);
        f->sub  = getf(m, 0, "%s.experts.sub_norm.weight", p);
        m->has_moe = 1;
        return;
    }
    f->up = getf(m, 0, "%s.gate_up.weight", p);  // fused at load, see fuse_slot()
    if (!f->up) return;
    f->down = getf(m, 1, "%s.down_proj.weight", p);
    f->sub  = getf(m, 0, "%s.sub_norm.weight", p);
}

// Index of a block's t_proj inside the fused "t_proj.all" matrix.
static int t_proj_index(model_t *m, const char *block_prefix) {
    char name[GGML_MAX_NAME];
    snprintf(name, sizeof name, "%s.t_proj.weight", block_prefix);
    for (int i = 0; i < m->n_tproj; i++) if (!strcmp(m->tproj_names[i], name)) return i;
    die("missing tensor %s", name);
    return -1;
}

// Append every "<prefix>.<i>" block that exists; returns the new count.
static int load_blocks(model_t *m, const char *prefix, block_t **list, int n) {
    for (int i = 0; getf(m, 0, "%s.%d.attn_norm.weight", prefix, i); i++) {
        *list = realloc(*list, sizeof(block_t) * (n + 1));
        block_t *b = &(*list)[n++];
        char p[GGML_MAX_NAME];
        snprintf(p, sizeof p, "%s.%d", prefix, i);
        b->attn_norm = getf(m, 1, "%s.attn_norm.weight", p);
        b->ffn_norm  = getf(m, 1, "%s.ffn_norm.weight", p);
        b->tidx      = t_proj_index(m, p);
        b->wqkv      = getf(m, 1, "%s.attn.qkv.weight", p);
        b->wo        = getf(m, 1, "%s.attn.o_proj.weight", p);
        b->q_norm    = getf(m, 0, "%s.attn.q_norm.weight", p);
        b->k_norm    = getf(m, 0, "%s.attn.k_norm.weight", p);
        b->attn_sub  = getf(m, 0, "%s.attn.sub_norm.weight", p);
        snprintf(p, sizeof p, "%s.%d.ffn", prefix, i);
        load_ffn(m, &b->ffn, p);
        if (!b->ffn.up) die("%s: no FFN weights", p);
    }
    return n;
}

static double cfg_num(struct gguf_context *g, const char *k, double def) {
    char key[128];
    snprintf(key, sizeof key, "yeoubi.%s", k);
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_INT32:   return gguf_get_val_i32(g, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(g, id);
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(g, id);
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(g, id);
        default:                return def;
    }
}

static const char *cfg_str(struct gguf_context *g, const char *k, const char *def) {
    char key[128];
    snprintf(key, sizeof key, "yeoubi.%s", k);
    int64_t id = gguf_find_key(g, key);
    return id < 0 ? def : gguf_get_val_str(g, id);
}

static float *host_copy(struct ggml_context *hctx, const char *name, int *n) {
    ten *t = ggml_get_tensor(hctx, name);
    if (!t) { *n = 0; return NULL; }
    *n = (int) ggml_nelements(t);
    float *out = malloc(ggml_nbytes(t));
    memcpy(out, t->data, ggml_nbytes(t));
    return out;
}

// supports_op, asked with the probe tensors placed in a real buffer on the
// device: Metal and the CPU's repacking buffer types decide partly by where
// a tensor lives, so asking about buffer-less tensors gave false negatives.
static int probe_ops(ggml_backend_t be, struct ggml_context *c, ten **ops, int n) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    int ok = buf != NULL;
    for (int i = 0; ok && i < n; i++) ok = ggml_backend_supports_op(be, ops[i]);
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(c);
    return ok;
}

static struct ggml_context *probe_ctx(void) {
    struct ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    return ggml_init(ip);
}

// Can the device run the fast TopK path (ggml_top_k + TOP_K_MASK) natively?
static int dev_supports_topk_mask(ggml_backend_t be) {
    struct ggml_context *c = probe_ctx();
    ten *a    = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2048, 64);
    ten *idx  = ggml_top_k(c, a, 1126);
    ten *mask = ggml_top_k_mask(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 2048, 1), idx, NULL, NULL, NULL);
    ten *ops[] = { idx, mask };
    return probe_ops(be, c, ops, 2);
}

// Does the device run the patched GGML_OP_ACT_QUANT natively, with and without a mask?
static int dev_supports_act_quant(ggml_backend_t be) {
    struct ggml_context *c = probe_ctx();
    ten *a    = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2048, 64);
    ten *mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2048, 64);
    ten *ops[] = { ggml_act_quant(c, a, NULL, 0), ggml_act_quant(c, a, mask, 2) };
    return probe_ops(be, c, ops, 2);
}

// Does the device run F32-precision flash attention with F32 K/V and no mask,
// on strided (permuted) q/k/v views, natively?
static int dev_supports_flash_attn(ggml_backend_t be, int hd, int H) {
    struct ggml_context *c = probe_ctx();
    ten *q = ggml_permute(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, H, 64), 0, 2, 1, 3);
    ten *k = ggml_permute(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, H, 64), 0, 2, 1, 3);
    ten *v = ggml_permute(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, H, 64), 0, 2, 1, 3);
    ten *o = ggml_flash_attn_ext(c, q, k, v, NULL, 1.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    return probe_ops(be, c, &o, 1);
}

// Widest row the device's fused TopK + INT8 (ACT_QUANT mode 3) takes, 0 if
// none: only the CPU has it; GPUs use TOP_K + TOP_K_MASK, which was faster.
static int dev_fused_topk_max(ggml_backend_t be) {
    static const int widths[] = { 65536 };
    for (size_t w = 0; w < sizeof widths / sizeof *widths; w++) {
        struct ggml_context *c = probe_ctx();
        ten *op = ggml_act_quant_topk(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, widths[w], 1), widths[w] / 2);
        if (probe_ops(be, c, &op, 1)) return widths[w];
    }
    return 0;
}

// Microseconds for one [k -> n_out] x 64-token matmul with weights of `type`
// on `be` (mean of 5 runs after a warm-up), or a huge value if the device
// has no native kernel for it.
static double time_matmul(ggml_backend_t be, enum ggml_type type, int k, int n_out) {
    struct ggml_context *c = probe_ctx();
    ten *w = ggml_new_tensor_2d(c, type, k, n_out);
    ten *x = ggml_new_tensor_2d(c, GGML_TYPE_F32, k, 64);
    ten *y = ggml_mul_mat(c, w, x);
    struct ggml_cgraph *gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    double us = 1e30;
    if (buf && ggml_backend_supports_op(be, y)) {
        ggml_backend_buffer_clear(buf, 0);
        ggml_backend_graph_compute(be, gf);  // warm-up (pipeline compile, caches)
        int64_t t0 = ggml_time_us();
        for (int i = 0; i < 5; i++) ggml_backend_graph_compute(be, gf);
        us = (ggml_time_us() - t0) / 5.0;
    }
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(c);
    return us;
}

// The exact in-memory type for ternary weights that is fastest here. F16 is
// tried on GPUs only, and only if it would use under half the free memory.
static enum ggml_type pick_weight_type(ggml_backend_t be, int is_cpu, int64_t tern_params, int hidden, int ffn) {
    enum ggml_type cand[] = { GGML_TYPE_Q4_0, GGML_TYPE_TQ2_0, GGML_TYPE_F16 };
    int n_cand = 3;
    if (is_cpu) {
        n_cand = 2;
    } else {
        size_t free_mem = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &free_mem, &total);
        if (free_mem && (size_t) tern_params * 2 > free_mem / 2) n_cand = 2;
    }
    if (hidden % 256) hidden = 768;  // TQ2_0 needs rows of 256
    enum ggml_type best = GGML_TYPE_Q4_0;
    double best_us = 1e30;
    fprintf(stderr, "weight type autotune:");
    for (int i = 0; i < n_cand; i++) {
        double us = time_matmul(be, cand[i], hidden, 2 * ffn);  // the fused gate/up shape
        if (us < 1e29) fprintf(stderr, " %s %.0f us", ggml_type_name(cand[i]), us);
        if (us < best_us) { best_us = us; best = cand[i]; }
    }
    fprintf(stderr, "\n");
    return best;
}

// TQ1_0 decoding table: byte -> its five base-3 digits, most significant
// first, exactly as gguf-py's TQ1_0.dequantize_blocks extracts them
// (multiply by 3^n mod 256, then * 3 >> 8). Codes 0/1/2 mean -1/0/+1.
static uint8_t tq1_trits[256][5];

static void tq1_init(void) {
    static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    for (int q = 0; q < 256; q++)
        for (int n = 0; n < 5; n++)
            tq1_trits[q][n] = (uint8_t) (((uint8_t) (q * pow3[n]) * 3) >> 8);
}

// One 256-weight TQ1_0 block (48 + 4 bytes of digits, then the fp16 scale) -> codes.
static void tq1_block_codes(const uint8_t *blk, uint8_t *t) {
    for (int m = 0; m < 32; m++) for (int n = 0; n < 5; n++) t[n * 32 + m]       = tq1_trits[blk[m]][n];
    for (int m = 0; m < 16; m++) for (int n = 0; n < 5; n++) t[160 + n * 16 + m] = tq1_trits[blk[32 + m]][n];
    for (int j = 0; j < 4; j++)  for (int n = 0; n < 4; n++) t[240 + n * 4 + j]  = tq1_trits[blk[48 + j]][n];
}

// TQ1_0 (the file's format) -> `to`, straight from the codes: every value is
// 0 or +/- the block's fp16 scale, so each target is written exactly without
// a float detour. (Through floats and ggml's quantizers this took ~1.6 s of
// a ~2 s startup for 106M weights.) Other targets take the float path.
static void *repack_ternary(const ten *src, enum ggml_type to, size_t *size) {
    int64_t n = ggml_nelements(src), cols = src->ne[0], rows = n / cols, nblk = n / 256;
    *size = ggml_row_size(to, cols) * rows;
    uint8_t *dst = malloc(*size);
    const uint8_t *in = (const uint8_t *) src->data;
    uint8_t t[256];
    if (to == GGML_TYPE_Q4_0 || to == GGML_TYPE_TQ2_0 || to == GGML_TYPE_F16) {
        for (int64_t blk = 0; blk < nblk; blk++, in += 54) {
            tq1_block_codes(in, t);
            const uint8_t *d = in + 52;  // fp16 scale, copied bit for bit
            if (to == GGML_TYPE_Q4_0) {
                // 8 Q4_0 blocks of {fp16 d, 16 bytes}: code j in the low nibble,
                // code j + 16 in the high one, value (nibble - 8) * d.
                uint8_t *o = dst + blk * 8 * 18;
                for (int s = 0; s < 8; s++, o += 18) {
                    memcpy(o, d, 2);
                    for (int j = 0; j < 16; j++)
                        o[2 + j] = (uint8_t) ((t[s * 32 + j] + 7) | ((t[s * 32 + j + 16] + 7) << 4));
                }
            } else if (to == GGML_TYPE_TQ2_0) {
                // {64 bytes, fp16 d}: element c*128 + n*32 + m in bits 2n of byte c*32 + m.
                uint8_t *o = dst + blk * 66;
                for (int c = 0; c < 2; c++)
                    for (int m = 0; m < 32; m++)
                        o[c * 32 + m] = (uint8_t) (t[c * 128 + m] | (t[c * 128 + 32 + m] << 2) |
                                                   (t[c * 128 + 64 + m] << 4) | (t[c * 128 + 96 + m] << 6));
                memcpy(o + 64, d, 2);
            } else {  // F16: 0, +d or -d
                uint16_t db, *o = (uint16_t *) dst + blk * 256;
                memcpy(&db, d, 2);
                for (int i = 0; i < 256; i++) o[i] = t[i] == 1 ? 0 : t[i] == 2 ? db : (uint16_t) (db ^ 0x8000u);
            }
        }
        return dst;
    }
    float *f = malloc(sizeof(float) * n);
    ggml_get_type_traits(src->type)->to_float(src->data, f, n);
    ggml_quantize_chunk(to, f, dst, 0, rows, cols, NULL);
    free(f);
    return dst;
}

// Projections that read the same input are fused into one matmul at load:
// q/k/v -> "<p>.qkv.weight", gate/up -> "<p>.gate_up.weight" (gate rows first,
// the order ggml_swiglu expects), and every block's t_proj -> "t_proj.all".
// At 64 patch positions per step the GPU is launch-bound, so one wide matmul
// beats three narrow ones. Returns the number of parts (0 = not fused).
static int fuse_slot(model_t *m, const char *name, const ten *src, char *fused, int *part) {
    static const struct { const char *suffix, *to; int part, parts; } rules[] = {
        { ".q_proj.weight",    ".qkv.weight",     0, 3 },
        { ".k_proj.weight",    ".qkv.weight",     1, 3 },
        { ".v_proj.weight",    ".qkv.weight",     2, 3 },
        { ".gate_proj.weight", ".gate_up.weight", 0, 2 },
        { ".up_proj.weight",   ".gate_up.weight", 1, 2 },
    };
    size_t n = strlen(name);
    if (strstr(name, ".t_proj.weight")) {
        for (int i = 0; i < m->n_tproj; i++) {
            if (!strcmp(m->tproj_names[i], name)) {
                *part = i;
                snprintf(fused, GGML_MAX_NAME, "t_proj.all");
                return m->n_tproj;
            }
        }
    }
    if (ggml_n_dims(src) != 2 || strstr(name, ".experts.")) return 0;  // MoE experts stay per projection
    for (size_t r = 0; r < sizeof rules / sizeof *rules; r++) {
        size_t s = strlen(rules[r].suffix);
        if (n > s && !strcmp(name + n - s, rules[r].suffix)) {
            snprintf(fused, GGML_MAX_NAME, "%.*s%s", (int) (n - s), name, rules[r].to);
            *part = rules[r].part;
            return rules[r].parts;
        }
    }
    return 0;
}

static void model_load(model_t *m, const char *path, int weight_type) {
    tq1_init();
    struct ggml_context *hctx = NULL;
    struct gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &hctx };
    struct gguf_context *g = gguf_init_from_file(path, gp);
    if (!g) die("cannot read GGUF %s", path);
    const char *type = cfg_str(g, "model_type", NULL);
    if (!type) die("%s is not a Yeoubi export (run: python -m yeoubi.gguf_export CKPT OUT.gguf)", path);
    if (cfg_num(g, "vocab_size", VOCAB) != VOCAB) die("not a byte-level checkpoint");
    if (cfg_num(g, "N_think", 0) > 0) die("legacy thinking-token checkpoints are not supported");

    m->is_rdt            = strcmp(type, "rdt") == 0;
    m->n_heads           = (int) cfg_num(g, "n_heads", 12);
    m->head_dim          = (int) cfg_num(g, "head_dim", 64);
    m->topk_ratio        = cfg_num(g, "topk_ratio", 0.55);
    m->int4_absmean      = (int) cfg_num(g, "int4_absmean", 0);
    m->rope_theta        = cfg_num(g, "rope_theta", 10000.0);
    m->rope_factor       = cfg_num(g, "rope_scaling_factor", 1.0);
    m->timestep_scale    = cfg_num(g, "timestep_scale", 1.0);
    m->max_loops         = (int) cfg_num(g, "max_loop_iters", 1);
    m->min_loops         = (int) cfg_num(g, "min_loop_iters", 1);
    m->adaptive_loops    = (int) cfg_num(g, "adaptive_loops", 0);
    m->context_adaptive  = (int) cfg_num(g, "context_adaptive_loops", 0);
    m->context_threshold = (int) cfg_num(g, "context_loop_threshold", 4096);
    m->state_std         = cfg_num(g, "recurrent_state_init_std", 0.0);
    const int cfg_use_act = (int) cfg_num(g, "use_act", 1);
    m->n_experts         = (int) cfg_num(g, "n_experts", 8);
    m->top_k             = (int) cfg_num(g, "top_k_experts", 2);
    m->capacity_factor   = cfg_num(g, "expert_capacity_factor", 1.25);
    m->kv_bits           = (int) cfg_num(g, "kv_cache_bits", 4);
    m->kv_bos_bits       = (int) cfg_num(g, "kv_cache_bos_bits", 4);
    const char *rope = cfg_str(g, "rope_scaling_type", "none");
    // Matches RotaryEmbedding._scaled: scaling only applies with factor > 1.
    m->rope_type = m->rope_factor <= 1.0       ? ROPE_NONE
                 : strcmp(rope, "linear") == 0 ? ROPE_LINEAR
                 : strcmp(rope, "yarn") == 0   ? ROPE_YARN : ROPE_NONE;
    if (m->top_k < 1 || m->top_k > MAX_TOPK) die("top_k_experts must be in [1, %d]", MAX_TOPK);
    if (m->kv_bits < 2 || m->kv_bits > 8 || m->kv_bos_bits < 2 || m->kv_bos_bits > 8) die("kv_cache bits out of range");

    // Ternary weights arrive as TQ1_0 (smallest on disk) and are repacked to
    // whichever exact type multiplies fastest on this device, timed now
    // rather than assumed: the best type differs by backend and CPU (TQ2_0
    // won on an x86 laptop CPU, Q4_0 on its Arc iGPU, ARM is different again).
    ggml_backend_dev_t dev = ggml_backend_get_device(m->backend);
    int is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    int64_t tern_params = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        ten *t = ggml_get_tensor(hctx, gguf_get_tensor_name(g, i));
        if (t->type == GGML_TYPE_TQ1_0) tern_params += ggml_nelements(t);
    }
    enum ggml_type wt = weight_type >= 0 ? (enum ggml_type) weight_type
                      : pick_weight_type(m->backend, is_cpu, tern_params,
                                         (int) cfg_num(g, "hidden_dim", 768), (int) cfg_num(g, "ffn_dim", 2048));
    // YEOUBI_NO_FAST_TOPK=1 forces the argsort path (for A/B checks).
    m->fast_topk = !getenv("YEOUBI_NO_FAST_TOPK") && dev_supports_topk_mask(m->backend);
    // YEOUBI_NO_FUSED_QUANT=1 forces the stock-op chains (for A/B checks).
    m->fused_quant = !getenv("YEOUBI_NO_FUSED_QUANT") && dev_supports_act_quant(m->backend);
    // YEOUBI_NO_FAST_TOPK=1 also turns this off, so CI can still reach the argsort path.
    m->fused_topk_max = m->fused_quant && !getenv("YEOUBI_NO_FAST_TOPK") ? dev_fused_topk_max(m->backend) : 0;
    m->views_compact = strncmp(ggml_backend_dev_name(dev), "OPENVINO", 8) == 0;
    // YEOUBI_NO_FLASH_ATTN=1 forces the matmul/softmax attention (for A/B checks).
    m->flash_attn = !getenv("YEOUBI_NO_FLASH_ATTN") &&
                    dev_supports_flash_attn(m->backend, (int) cfg_num(g, "head_dim", 64), (int) cfg_num(g, "n_heads", 12));
    fprintf(stderr, "topk: %s, act quant: %s, attention: %s\n",
            m->fused_topk_max ? "fused" : m->fast_topk ? "top_k + top_k_mask" : "argsort",
            m->fused_quant ? "fused" : "op chain", m->flash_attn ? "flash" : "matmul");

    // Weights: mirror every GGUF tensor into one buffer on the chosen device,
    // fusing projections that read the same input (see fuse_slot).
    int64_t nt = gguf_get_n_tensors(g);
    struct ggml_init_params ip = { (size_t) (nt + 8) * ggml_tensor_overhead(), NULL, true };
    m->wctx = ggml_init(ip);
    m->tproj_names = malloc(sizeof *m->tproj_names * (size_t) nt);
    for (int64_t i = 0; i < nt; i++) {
        const char *name = gguf_get_tensor_name(g, i);
        if (strstr(name, ".t_proj.weight")) snprintf(m->tproj_names[m->n_tproj++], GGML_MAX_NAME, "%s", name);
    }
    int64_t n_tern = 0;
    for (int64_t i = 0; i < nt; i++) {
        const char *name = gguf_get_tensor_name(g, i);
        ten *src = ggml_get_tensor(hctx, name);
        enum ggml_type t = src->type == GGML_TYPE_TQ1_0 ? wt : src->type;
        if (src->type == GGML_TYPE_TQ1_0) n_tern += ggml_nelements(src);
        char fused[GGML_MAX_NAME];
        int part, parts = fuse_slot(m, name, src, fused, &part);
        if (!parts) {
            ggml_set_name(ggml_new_tensor(m->wctx, t, ggml_n_dims(src), src->ne), name);
        } else if (!get(m, fused)) {
            ggml_set_name(ggml_new_tensor_2d(m->wctx, t, src->ne[0], src->ne[1] * parts), fused);
        }
    }
    m->wbuf = ggml_backend_alloc_ctx_tensors_from_buft(m->wctx, ggml_backend_get_default_buffer_type(m->backend));
    if (!m->wbuf) die("out of device memory for weights");
    for (int64_t i = 0; i < nt; i++) {
        const char *name = gguf_get_tensor_name(g, i);
        ten *src = ggml_get_tensor(hctx, name);
        char fused[GGML_MAX_NAME];
        int part = 0, parts = fuse_slot(m, name, src, fused, &part);
        ten *dst = get(m, parts ? fused : name);
        // A fused part is a run of whole rows: part p starts p row-blocks in.
        size_t offset = (size_t) part * ggml_row_size(dst->type, src->ne[0]) * src->ne[1];
        if (dst->type == src->type) {
            ggml_backend_tensor_set(dst, src->data, offset, ggml_nbytes(src));
        } else {
            size_t size;
            void *buf = repack_ternary(src, dst->type, &size);
            ggml_backend_tensor_set(dst, buf, offset, size);
            free(buf);
        }
    }
    m->tproj_all = get(m, "t_proj.all");
    if (!m->tproj_all) die("no t_proj weights in %s", path);
    fprintf(stderr, "weights: %.1fM ternary as %s, %.1f MB on device\n", n_tern / 1e6,
            ggml_type_name(wt), ggml_backend_buffer_get_size(m->wbuf) / 1e6);
    m->alpha      = host_copy(hctx, "recurrent.alpha", &m->n_alpha);
    m->iter_gate  = host_copy(hctx, "recurrent.lora.iter_gate", &m->n_gate);
    m->loop_embed = host_copy(hctx, "recurrent.loop_embed.weight", &m->n_loop_embed);
    ggml_free(hctx);
    gguf_free(g);

    m->embed  = need(m, "embed.weight");
    m->head   = get(m, "unembed.weight");  // only written when tie_embeddings is off
    if (!m->head) m->head = m->embed;
    m->hidden = (int) m->embed->ne[0];
    m->ne0_w  = need(m, "noise_embed.proj.0.weight");
    m->ne0_b  = need(m, "noise_embed.proj.0.bias");
    m->ne2_w  = need(m, "noise_embed.proj.2.weight");
    m->ne2_b  = need(m, "noise_embed.proj.2.bias");
    m->t_dim  = (int) m->ne0_w->ne[0];
    m->final_norm     = need(m, "final_norm.weight");
    m->self_cond_proj = get(m, "self_cond_proj.weight");

    m->subpos = get(m, "byte_codec.subposition");
    m->patch  = m->subpos ? (int) m->subpos->ne[1] : 1;
    if (m->subpos) {
        m->enc_norm     = need(m, "byte_codec.encoder_norm.weight");
        m->enc_conv     = need(m, "byte_codec.encoder_local.weight");
        m->cmp_norm     = need(m, "byte_codec.compress_norm.weight");
        m->cmp          = need(m, "byte_codec.compress.weight");
        m->exp          = need(m, "byte_codec.expand.weight");
        m->dec_norm     = need(m, "byte_codec.decoder_norm.weight");
        m->dec_conv     = need(m, "byte_codec.decoder_local.weight");
        m->dec_ffn_norm = get(m, "byte_codec.decoder_ffn_norm.weight");
        m->out_norm     = need(m, "byte_codec.output_norm.weight");
        load_ffn(m, &m->dec_ffn, "byte_codec.decoder_ffn");
        if (m->enc_conv->ne[1] != m->hidden) die("old GGUF conv layout: re-export with this version");
    }

    if (m->is_rdt) {
        m->n_pre  = load_blocks(m, "prelude", &m->pre, 0);
        m->n_pre  = load_blocks(m, "pre_recurrent", &m->pre, m->n_pre);
        m->n_rec  = load_blocks(m, "recurrent.shared_blocks", &m->rec, 0);
        m->n_post = load_blocks(m, "post_recurrent", &m->post, 0);
        m->n_post = load_blocks(m, "coda", &m->post, m->n_post);
        m->inj_A        = need(m, "recurrent.injection.A");
        m->inj_B        = need(m, "recurrent.injection.B.weight");
        m->lora_down    = need(m, "recurrent.lora.down.weight");
        m->lora_up      = need(m, "recurrent.lora.up.weight");
        // BitRecurrentBlock builds the ACT head whatever use_act says, so the
        // weights alone do not mean the blend is on: the config decides.
        m->act_w        = cfg_use_act ? get(m, "recurrent.act.weight_proj.weight") : NULL;
        m->loop_proj    = need(m, "recurrent.loop_proj.weight");
        m->loop_proj2   = need(m, "recurrent.loop_proj2.weight");
        m->loop_proj2_b = need(m, "recurrent.loop_proj2.bias");
        m->loop_norm    = get(m, "recurrent.loop_norm.weight");
        while (m->n_loop_norms < MAX_LOOPS && getf(m, 0, "recurrent.loop_norms.%d.weight", m->n_loop_norms)) {
            m->loop_norms[m->n_loop_norms] = getf(m, 1, "recurrent.loop_norms.%d.weight", m->n_loop_norms);
            m->n_loop_norms++;
        }
        m->loop_dim = (int) m->loop_proj->ne[1];
        if (!m->alpha || !m->iter_gate) die("missing recurrent.alpha / lora.iter_gate");
        if (!m->n_rec) die("RDT export has no recurrent.shared_blocks");
    } else {
        m->n_pre = load_blocks(m, "blocks", &m->pre, 0);
    }
}

// KV cache sized for `loops` recurrent iterations and `cap` global positions.
static void kv_alloc(model_t *m, int loops, int cap) {
    m->kv_loops  = loops;
    m->kv_layers = m->n_pre + m->n_post + (m->is_rdt ? m->n_rec * loops : 0);
    m->kv_cap    = cap;
    m->kv_len    = 0;
    struct ggml_init_params ip = { (size_t) 2 * m->kv_layers * ggml_tensor_overhead(), NULL, true };
    m->kvctx = ggml_init(ip);
    m->kv_k = malloc(sizeof(ten *) * m->kv_layers);
    m->kv_v = malloc(sizeof(ten *) * m->kv_layers);
    for (int i = 0; i < m->kv_layers; i++) {
        // ponytail: stored dequantized in F32 (exactly PyTorch's read-back
        // values) rather than bit-packed; pack when long contexts need the memory.
        m->kv_k[i] = ggml_new_tensor_3d(m->kvctx, GGML_TYPE_F32, m->head_dim, m->n_heads, cap);
        m->kv_v[i] = ggml_new_tensor_3d(m->kvctx, GGML_TYPE_F32, m->head_dim, m->n_heads, cap);
    }
    m->kvbuf = ggml_backend_alloc_ctx_tensors_from_buft(m->kvctx, ggml_backend_get_default_buffer_type(m->backend));
    if (!m->kvbuf) die("out of device memory for the KV cache");
    ggml_backend_buffer_clear(m->kvbuf, 0);
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

typedef struct {
    struct ggml_context *ctx;
    struct ggml_cgraph *gf;
    model_t *m;
    ten *rope_cos, *rope_sin, *consts, *eye, *tril, *tbias, *zeros;
    float cvals[N_CONST];
    int ncon, kv, layer;
} gb_t;

// A [1] view of a host-filled constant table. Views of one input keep every
// scalar on the device; deriving them from a 1-element input with ggml_scale
// put that op (and a sync) on the CPU.
static ten *konst(gb_t *b, float v) {
    int i = 0;
    while (i < b->ncon && b->cvals[i] != v) i++;
    if (i == b->ncon) {
        if (b->ncon == N_CONST) die("constant table full");
        b->cvals[b->ncon++] = v;
    }
    return ggml_view_1d(b->ctx, b->consts, 1, (size_t) i * sizeof(float));
}

static ten *norm(gb_t *b, ten *x, ten *w) {
    if (!w) return x;  // nn.Identity in the PyTorch model
    return ggml_mul(b->ctx, ggml_rms_norm(b->ctx, x, 1e-6f), w);
}

// a: contiguous [n, rows]; idx: I32 with one index per row -> [1, rows].
static ten *pick(gb_t *b, ten *a, ten *idx) {
    struct ggml_context *c = b->ctx;
    int64_t n = a->ne[0], rows = a->ne[1];
    // Each element becomes a 1-wide row, so get_rows gathers one value per row.
    ten *v = ggml_get_rows(c, ggml_reshape_3d(c, a, 1, n, rows), ggml_reshape_2d(c, idx, 1, rows));
    return ggml_reshape_2d(c, v, 1, rows);
}

// Per-row max of a contiguous [n, rows] tensor -> [1, rows], as one max-pool
// over the whole row. argmax + gather cost two ops and a slow 1-wide gather.
static ten *row_max(gb_t *b, ten *a) {
    int n = (int) a->ne[0];
    return ggml_pool_2d(b->ctx, a, GGML_OP_POOL_MAX, n, 1, n, 1, 0.0f, 0.0f);
}

// HybridQuantizer.forward / absmax_quantize_int8: per-token absmax, +/-127.
// `amax` may be passed when the caller already has it.
static ten *q_int8(gb_t *b, ten *x, ten *amax) {
    struct ggml_context *c = b->ctx;
    if (!amax) amax = row_max(b, ggml_abs(c, x));
    ten *step = ggml_scale(c, ggml_clamp(c, amax, 1e-8f, FLT_MAX), 1.0f / 127.0f);
    return ggml_mul(c, ggml_clamp(c, ggml_round(c, ggml_div(c, x, step)), -127.0f, 127.0f), step);
}

// HybridQuantizer "int4": absmean (a4.8) or absmax per-token scale, +/-7.
static ten *q_int4(gb_t *b, ten *x) {
    struct ggml_context *c = b->ctx;
    ten *a = ggml_abs(c, x), *step;
    if (b->m->int4_absmean) step = ggml_scale(c, ggml_clamp(c, ggml_mean(c, a), 1e-8f, FLT_MAX), 1.0f / sqrtf(7.0f));
    else                    step = ggml_scale(c, ggml_clamp(c, row_max(b, a), 1e-8f, FLT_MAX), 1.0f / 7.0f);
    return ggml_mul(c, ggml_clamp(c, ggml_round(c, ggml_div(c, x, step)), -7.0f, 7.0f), step);
}

// HybridQuantizer "topk_int8": keep the int(n * ratio) largest |x| per token,
// then INT8 with the row's absmax (the largest survives TopK).
//
// Fast path, where the device has both ops: ggml_top_k (radix select) picks
// exactly k indices, TOP_K_MASK turns them into a dense 0 / -inf row and exp
// makes that 1 / 0. Vulkan gets TOP_K_MASK from runtime/cmake/patch_ggml.cmake.
// Fallback: one descending argsort, whose column k-1 is the threshold.
// (ggml_top_k's indices are unordered, so reading the threshold from them
// would need a k-wide gather, which on Vulkan cost a third of each step.)
// ponytail: ggml_round rounds halves away from zero, torch to even; exact
// .5 products are ~1 in 10^6 values and move one value by one level.
static ten *q_topk8(gb_t *b, ten *x) {
    struct ggml_context *c = b->ctx;
    int64_t n = x->ne[0], rows = x->ne[1];
    int k = (int) (n * b->m->topk_ratio);
    if (k < 1) k = 1;
    int fused = b->m->fused_quant;
    if (k >= n) return fused ? ggml_act_quant(c, x, NULL, 2) : q_int8(b, x, NULL);
    if (n <= b->m->fused_topk_max) return ggml_act_quant_topk(c, x, k);  // select + drop + INT8, one kernel
    ten *a = ggml_abs(c, x);
    if (b->m->fast_topk && n <= ZEROS_MAX) {
        ten *zeros = ggml_view_2d(c, b->zeros, n, 1, (size_t) n * sizeof(float), 0);
        ten *mask  = ggml_top_k_mask(c, zeros, ggml_top_k(c, a, k), NULL, NULL, NULL);
        if (fused) return ggml_act_quant(c, x, mask, 2);  // drop + INT8 in one kernel
        return q_int8(b, ggml_mul(c, x, ggml_exp(c, mask)), row_max(b, a));
    }
    ten *order = ggml_argsort(c, a, GGML_SORT_ORDER_DESC);
    ten *col0  = ggml_cont(c, ggml_view_2d(c, order, 1, rows, order->nb[1], 0));
    ten *colk  = ggml_cont(c, ggml_view_2d(c, order, 1, rows, order->nb[1], (size_t) (k - 1) * order->nb[0]));
    ten *thr   = pick(b, a, colk);
    // 1 - 1e-6 keeps the k-th value itself (step(0) == 0).
    x = ggml_mul(c, x, ggml_step(c, ggml_sub(c, a, ggml_scale(c, thr, 1.0f - 1e-6f))));
    return fused ? ggml_act_quant(c, x, NULL, 2) : q_int8(b, x, pick(b, a, col0));
}

// Activation fake-quant for a BitLinear input. Where the device has the
// patched GGML_OP_ACT_QUANT (CPU, Vulkan, Metal), each quantizer is one
// kernel; elsewhere (CUDA/HIP replay whole graphs as CUDA graphs, so the op
// chains cost them little) the equivalent chain of stock ops.
static ten *quant(gb_t *b, ten *x, int mode) {
    model_t *m = b->m;
    if (m->act == ACT_NONE) return x;
    if (m->act == ACT_A8 || mode == Q_INT8)  // set_activation_mode("A8"): INT8 everywhere
        return m->fused_quant ? ggml_act_quant(b->ctx, x, NULL, 2) : q_int8(b, x, NULL);
    if (mode == Q_TOPK) return q_topk8(b, x);
    return m->fused_quant ? ggml_act_quant(b->ctx, x, NULL, m->int4_absmean ? 0 : 1) : q_int4(b, x);
}

static ten *lin(gb_t *b, ten *w, ten *x, int mode) { return ggml_mul_mat(b->ctx, w, quant(b, x, mode)); }

static ten *ffn(gb_t *b, ffn_t *f, ten *h) {
    struct ggml_context *c = b->ctx;
    h = quant(b, h, Q_INT4);  // up and gate share one quantized input
    ten *mid = ggml_swiglu(c, ggml_mul_mat(c, f->up, h));  // silu(gate) * up from the fused rows
    return lin(b, f->down, norm(b, mid, f->sub), Q_TOPK);
}

// BitMoEFFN.forward (eager path, which sample.py runs): softmax router,
// top-k, renormalise, Switch capacity, probability-weighted expert sum.
static ten *moe(gb_t *b, ffn_t *f, ten *h) {
    struct ggml_context *c = b->ctx;
    model_t *m = b->m;
    int N = m->n_experts, K = m->top_k;
    int64_t D = h->ne[0], T = h->ne[1];
    if (f->up->ne[2] != N) die("MoE: %d experts in weights, %d in config", (int) f->up->ne[2], N);

    // Router: Int8Linear, its weight already int8-dequantized at export.
    ten *probs = ggml_soft_max(c, ggml_mul_mat(c, f->router, h));       // [N, T]
    ten *order = ggml_argsort(c, probs, GGML_SORT_ORDER_DESC);          // [N, T]
    ten *ids   = ggml_cont(c, ggml_view_2d(c, order, K, T, order->nb[1], 0));  // [K, T]

    // Capacity in the eager loop's order (slot-major, then token order):
    // pick (k, t) of expert e survives iff picks of e in earlier slots plus
    // earlier tokens of slot k are < capacity. Counts come from one-hot rows
    // of an identity matrix and a strictly-lower-triangular matmul.
    int cap = (int) ((double) T / N * m->capacity_factor);
    if (cap < 1) cap = 1;
    ten *seen = NULL, *psum = NULL, *p[MAX_TOPK], *keep[MAX_TOPK];
    for (int k = 0; k < K; k++) {
        ten *idk  = ggml_cont(c, ggml_view_2d(c, order, 1, T, order->nb[1], (size_t) k * order->nb[0]));
        ten *hot  = ggml_get_rows(c, b->eye, ggml_reshape_1d(c, idk, T));       // [N, T]
        ten *hotT = ggml_cont(c, ggml_transpose(c, hot));                         // [T, N]
        ten *rank = ggml_mul_mat(c, hotT, b->tril);                               // [N, T]
        if (seen) rank = ggml_add(c, rank, seen);
        ten *ok = ggml_step(c, ggml_add(c, ggml_scale(c, rank, -1.0f), konst(b, (float) cap - 0.5f)));
        keep[k] = ggml_sum_rows(c, ggml_mul(c, hot, ok));                         // [1, T]
        p[k]    = ggml_sum_rows(c, ggml_mul(c, hot, probs));                      // [1, T]
        ten *cnt = ggml_reshape_2d(c, ggml_sum_rows(c, hotT), N, 1);
        seen = seen ? ggml_add(c, seen, cnt) : cnt;
        psum = psum ? ggml_add(c, psum, p[k]) : p[k];
    }
    ten *w = NULL;  // [1, K, T]
    for (int k = 0; k < K; k++) {
        ten *wk = ggml_reshape_3d(c, ggml_mul(c, ggml_div(c, p[k], psum), keep[k]), 1, 1, T);
        w = w ? ggml_concat(c, w, wk, 1) : wk;
    }

    ten *x3 = ggml_reshape_3d(c, quant(b, h, Q_INT4), D, 1, T);
    ten *mid = ggml_mul(c, ggml_silu(c, ggml_mul_mat_id(c, f->gate, x3, ids)),
                        ggml_mul_mat_id(c, f->up, x3, ids));                      // [F, K, T]
    int64_t F = mid->ne[0];
    if (f->sub) {  // per-expert ffn_sub_norm gain
        ten *g = ggml_get_rows(c, f->sub, ggml_reshape_1d(c, ids, (int64_t) K * T));
        mid = ggml_mul(c, ggml_rms_norm(c, mid, 1e-6f), ggml_reshape_3d(c, g, F, K, T));
    }
    mid = ggml_reshape_3d(c, quant(b, ggml_reshape_2d(c, mid, F, (int64_t) K * T), Q_TOPK), F, K, T);
    ten *out = ggml_mul(c, ggml_mul_mat_id(c, f->down, mid, ids), w);           // [D, K, T]
    ten *sum = NULL;
    for (int k = 0; k < K; k++) {
        ten *v = ggml_view_2d(c, out, D, T, out->nb[2], (size_t) k * out->nb[1]);
        sum = sum ? ggml_add(c, sum, v) : ggml_cont(c, v);
    }
    return sum;
}

// RotaryEmbedding.forward: x * cos + rotate_half(x) * sin with x = [hd, H, T].
// rotate_half(x) = cat(-x2, x1) = roll(x, hd/2) * [-1.., +1..]; the sign is
// folded into the sin table. Host tables, not ggml_rope_ext: they reproduce
// PyTorch's YaRN ramp and attention temperature exactly, and the native op
// with YaRN frequency factors had been landing on the CPU under Vulkan.
static ten *rope(gb_t *b, ten *x) {
    struct ggml_context *c = b->ctx;
    ten *rot = ggml_roll(c, x, b->m->head_dim / 2, 0, 0, 0);
    return ggml_add(c, ggml_mul(c, x, b->rope_cos), ggml_mul(c, rot, b->rope_sin));
}

// cos and signed sin [hd, Tp] for positions off.., as RotaryEmbedding._build_cache.
static void rope_tables(model_t *m, int off, int Tp, float *cos_t, float *sin_t) {
    int hd = m->head_dim, n = hd / 2;
    float s = (float) m->rope_factor;
    float mscale = m->rope_type == ROPE_YARN ? (float) (0.1 * log(m->rope_factor) + 1.0) : 1.0f;
    for (int i = 0; i < n; i++) {
        float inv = 1.0f / powf((float) m->rope_theta, (float) (2 * i) / (float) hd);
        if (m->rope_type == ROPE_YARN) {
            float ramp = n > 1 ? (float) i / (float) (n - 1) : 0.0f;
            ramp = fminf(fmaxf((ramp - 0.25f) / 0.5f, 0.0f), 1.0f);
            inv *= (1.0f - ramp) + ramp / s;
        }
        for (int p = 0; p < Tp; p++) {
            float pos = (float) (off + p);
            float a = (m->rope_type == ROPE_LINEAR ? pos / s : pos) * inv;
            float cv = cosf(a) * mscale, sv = sinf(a) * mscale;
            cos_t[(size_t) p * hd + i] = cos_t[(size_t) p * hd + i + n] = cv;
            sin_t[(size_t) p * hd + i] = -sv;
            sin_t[(size_t) p * hd + i + n] = sv;
        }
    }
}

// KVCache quantize_kv + dequantize_kv: per (token, head) absmax over head_dim.
static ten *kv_quant_bits(gb_t *b, ten *x, int bits) {
    struct ggml_context *c = b->ctx;
    int64_t hd = x->ne[0], H = x->ne[1], T = x->ne[2];
    float half = (float) (((1 << bits) - 1) / 2), top = (float) ((1 << bits) - 1);
    ten *x2 = ggml_reshape_2d(c, ggml_cont(c, x), hd, H * T);
    ten *scale = ggml_div(c, ggml_clamp(c, row_max(b, ggml_abs(c, x2)), 1e-8f, FLT_MAX), konst(b, half));
    ten *q = ggml_clamp(c, ggml_round(c, ggml_add(c, ggml_div(c, x2, scale), konst(b, half))), 0.0f, top);
    return ggml_reshape_3d(c, ggml_mul(c, ggml_add(c, q, konst(b, -half)), scale), hd, H, T);
}

static ten *kv_quant(gb_t *b, ten *x, int first) {
    model_t *m = b->m;
    int64_t T = x->ne[2];
    // KVCache.update stores the first position ever (legacy "BOS") at its own width.
    if (first && m->kv_bos_bits != m->kv_bits && T > 1) {
        ten *a = ggml_view_3d(b->ctx, x, x->ne[0], x->ne[1], 1, x->nb[1], x->nb[2], 0);
        ten *r = ggml_view_3d(b->ctx, x, x->ne[0], x->ne[1], T - 1, x->nb[1], x->nb[2], x->nb[2]);
        return ggml_concat(b->ctx, kv_quant_bits(b, a, m->kv_bos_bits), kv_quant_bits(b, r, m->kv_bits), 2);
    }
    return kv_quant_bits(b, x, first ? m->kv_bos_bits : m->kv_bits);
}

static ten *kv_view(gb_t *b, ten *t, int start, int64_t n) {
    return ggml_view_3d(b->ctx, t, t->ne[0], t->ne[1], n, t->nb[1], t->nb[2], (size_t) start * t->nb[2]);
}

static ten *attention(gb_t *b, block_t *L, ten *h) {
    struct ggml_context *c = b->ctx;
    model_t *m = b->m;
    int hd = m->head_dim, H = m->n_heads;
    int64_t T = h->ne[1];
    h = quant(b, h, Q_INT4);  // q, k, v share one quantized input
    ten *qkv = ggml_mul_mat(c, L->wqkv, h);  // [3 * H * hd, T]
    size_t es = ggml_element_size(qkv), part = (size_t) hd * H * es;
    ten *qv = ggml_view_3d(c, qkv, hd, H, T, hd * es, qkv->nb[1], 0);
    ten *kw = ggml_view_3d(c, qkv, hd, H, T, hd * es, qkv->nb[1], part);
    ten *vv = ggml_view_3d(c, qkv, hd, H, T, hd * es, qkv->nb[1], 2 * part);
    // RMS norm reads the strided views directly (its rows are contiguous) and
    // writes a compact tensor, so q/k only need a copy without q/k-norm. v is
    // consumed through a permute anyway, except by the cache writes.
    // (Not on OpenVINO: its translator treats VIEW as a no-op, so every split
    // is made compact there.)
    const int compact = m->views_compact;
    ten *q = rope(b, L->q_norm && !compact ? norm(b, qv, L->q_norm) : norm(b, ggml_cont(c, qv), L->q_norm));
    ten *k = rope(b, L->k_norm && !compact ? norm(b, kw, L->k_norm) : norm(b, ggml_cont(c, kw), L->k_norm));
    ten *v = b->kv != KV_NONE || compact ? ggml_cont(c, vv) : vv;

    int li = b->layer++;
    if (b->kv != KV_NONE) {
        if (li >= m->kv_layers) die("KV cache has %d layers, graph needs more", m->kv_layers);
        int L0 = m->kv_len;
        if (b->kv == KV_COMMIT) {
            // Committing attends over the cache's own dequantized copy, as
            // KVCache.update returns it, and writes that copy back.
            k = kv_quant(b, k, L0 == 0);
            v = kv_quant(b, v, L0 == 0);
            ggml_build_forward_expand(b->gf, ggml_cpy(c, k, kv_view(b, m->kv_k[li], L0, T)));
            ggml_build_forward_expand(b->gf, ggml_cpy(c, v, kv_view(b, m->kv_v[li], L0, T)));
        }
        if (L0 > 0) {
            k = ggml_concat(c, kv_view(b, m->kv_k[li], 0, L0), k, 2);
            v = ggml_concat(c, kv_view(b, m->kv_v[li], 0, L0), v, 2);
        }
    }

    // Bidirectional and unpadded (the cache holds no padding): no mask at all.
    const float scale = 1.0f / sqrtf((float) hd);
    q = ggml_permute(c, q, 0, 2, 1, 3);                 // [hd, T, H]
    k = ggml_permute(c, k, 0, 2, 1, 3);                 // [hd, Tk, H]
    ten *o;
    if (m->flash_attn) {
        // One fused kernel instead of QK^T, softmax, PV and two copies, at F32
        // precision so it matches the unfused path; output is already [hd, H, T].
        o = ggml_flash_attn_ext(c, q, k, ggml_permute(c, v, 0, 2, 1, 3), NULL, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
    } else {
        v = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));  // [Tk, hd, H]
        ten *kq = ggml_soft_max_ext(c, ggml_mul_mat(c, k, q), NULL, scale, 0.0f);
        o = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, v, kq), 0, 2, 1, 3));  // [hd, H, T]
    }
    o = ggml_reshape_2d(c, o, (int64_t) hd * H, T);
    return lin(b, L->wo, norm(b, o, L->attn_sub), Q_TOPK);
}

static ten *block(gb_t *b, block_t *L, ten *x, ten *temb) {
    struct ggml_context *c = b->ctx;
    (void) temb;  // its per-block projections were all done at once (b->tbias)
    ten *tb = ggml_view_1d(c, b->tbias, b->m->hidden, (size_t) L->tidx * b->m->hidden * ggml_element_size(b->tbias));
    ten *h = ggml_add(c, norm(b, x, L->attn_norm), tb);
    x = ggml_add(c, x, attention(b, L, h));
    h = norm(b, x, L->ffn_norm);
    return ggml_add(c, x, L->ffn.router ? moe(b, &L->ffn, h) : ffn(b, &L->ffn, h));
}

// BytePatchCodec._local_residual: x + depthwise_conv(norm(x)), zero padded
// on both sides. ggml_ssm_conv is exactly that depthwise cross-correlation
// over a pre-padded input, in one op instead of K multiply-adds.
static ten *local_residual(gb_t *b, ten *x, ten *nw, ten *w /* [K, D] */) {
    struct ggml_context *c = b->ctx;
    ten *l = norm(b, x, nw);
    int64_t D = l->ne[0], T = l->ne[1];
    int K = (int) w->ne[0];
    ten *lt = ggml_pad_ext(c, ggml_cont(c, ggml_transpose(c, l)), K / 2, K / 2, 0, 0, 0, 0, 0, 0);
    ten *y = ggml_ssm_conv(c, ggml_reshape_3d(c, lt, lt->ne[0], D, 1), w);  // [D, T, 1]
    return ggml_add(c, x, ggml_reshape_2d(c, y, D, T));
}

static int clampi(int i, int n) { return i < n ? i : n - 1; }

// Inputs a given model never reads are pruned from the graph and get no
// buffer; writing to them would assert.
static void set_input(ten *t, const void *data, size_t size) {
    if (t && t->buffer) ggml_backend_tensor_set(t, data, 0, size);
}

static ten *recurrent(gb_t *b, ten *x, ten *temb, int n_loops, ten *noise, ten *sig) {
    struct ggml_context *c = b->ctx;
    model_t *m = b->m;
    // Loop invariants: B(e) and the timestep projections never change per loop.
    ten *Be = lin(b, m->inj_B, x, Q_INT8);
    ten *ts = ggml_mul_mat(c, m->loop_proj, temb);
    ten *tb = ggml_add(c, ggml_mul_mat(c, m->loop_proj2, temb), m->loop_proj2_b);
    ten *h  = noise ? ggml_add(c, x, ggml_scale(c, noise, (float) m->state_std)) : x;
    ten *hs[MAX_LOOPS], *ws[MAX_LOOPS];

    for (int i = 0; i < n_loops; i++) {
        ten *s = ggml_view_2d(c, sig, m->loop_dim, 1, sig->nb[1], (size_t) i * sig->nb[1]);
        ten *delta = ggml_add(c, ggml_mul(c, s, ts), tb);
        h = ggml_add(c, h, ggml_pad(c, delta, m->hidden - m->loop_dim, 0, 0, 0));

        ten *ln = m->n_loop_norms ? m->loop_norms[clampi(i, m->n_loop_norms)] : m->loop_norm;
        ten *bo = norm(b, h, ln);
        for (int r = 0; r < m->n_rec; r++) bo = block(b, &m->rec[r], bo, temb);
        ten *lora = lin(b, m->lora_up, lin(b, m->lora_down, bo, Q_INT8), Q_TOPK);
        bo = ggml_add(c, bo, ggml_scale(c, lora, m->iter_gate[clampi(i, m->n_gate)]));
        bo = ggml_scale(c, bo, m->alpha[clampi(i, m->n_alpha)]);
        h = ggml_add(c, ggml_add(c, ggml_mul(c, h, m->inj_A), Be), bo);
        hs[i] = h;
        if (m->act_w) ws[i] = ggml_sigmoid(c, lin(b, m->act_w, h, Q_INT8));  // [1, T]
    }
    if (!m->act_w) return h;

    // Soft ACT blend (BitRecurrentBlock.forward): normalise weights across
    // loops, sequential lerp, divide out the achieved total weight.
    ten *one = konst(b, 1.0f);
    ten *sum = ws[0];
    for (int i = 1; i < n_loops; i++) sum = ggml_add(c, sum, ws[i]);
    sum = ggml_add(c, sum, konst(b, 1e-6f));
    ten *out = NULL, *keep = NULL;
    for (int i = 0; i < n_loops; i++) {
        ten *wn = ggml_div(c, ws[i], sum);
        out = out ? ggml_add(c, out, ggml_mul(c, ggml_sub(c, hs[i], out), wn)) : ggml_mul(c, hs[i], wn);
        ten *rest = ggml_add(c, ggml_scale(c, wn, -1.0f), one);  // 1 - wn
        keep = keep ? ggml_mul(c, keep, rest) : rest;
    }
    ten *total = ggml_clamp(c, ggml_add(c, ggml_scale(c, keep, -1.0f), one), 1e-6f, FLT_MAX);
    return ggml_div(c, out, total);
}

// One denoiser call. ids: T byte IDs (T % patch == 0). sc: optional
// self-conditioning probabilities [T][VOCAB]. out: logits [T][embed rows]
// (unused for KV_COMMIT). kv: KV_NONE (full canvas), KV_READ (attend over the
// committed cache), KV_COMMIT (store this canvas at t = 0).
// The last built graph. A denoising loop calls forward() with the same shape
// every step, so the graph is built and scheduled once and only its inputs
// change: rebuilding cost several ms of host time per step.
static struct {
    struct ggml_context *ctx;
    gb_t b;
    ten *in_ids, *in_tf, *in_sc, *in_sig, *in_noise, *logits;
    int T, loops, kv, kv_len, has_sc, has_noise;
} G;

static void build_graph(model_t *m, int T, int n_loops, int has_sc, int has_noise, int kv) {
    int D = m->hidden, P = m->patch, Tp = T / P, hd = m->head_dim;
    int visits = m->n_pre + m->n_post + (m->is_rdt ? m->n_rec * n_loops : 0);
    size_t graph_size = (size_t) 320 * (visits + 4) + 4096;  // nodes; ~200 per block visit with MoE
    size_t mem = ggml_tensor_overhead() * graph_size * 2 + ggml_graph_overhead_custom(graph_size, false);
    struct ggml_init_params ip = { mem, NULL, true };
    if (G.ctx) ggml_free(G.ctx);
    memset(&G, 0, sizeof G);
    struct ggml_context *c = G.ctx = ggml_init(ip);
    G.T = T; G.loops = n_loops; G.kv = kv; G.kv_len = m->kv_len; G.has_sc = has_sc; G.has_noise = has_noise;
    gb_t *bp = &G.b;
    gb_t b = { 0 };
    b.ctx = c;
    b.m = m;
    b.kv = kv;
    b.gf = ggml_new_graph_custom(c, graph_size, false);

    ten *in_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ten *in_tf  = ggml_new_tensor_2d(c, GGML_TYPE_F32, m->t_dim, 1);
    b.rope_cos = ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, 1, Tp);  // broadcast over heads
    b.rope_sin = ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, 1, Tp);
    b.consts   = ggml_new_tensor_1d(c, GGML_TYPE_F32, N_CONST);
    if (m->fast_topk) b.zeros = ggml_new_tensor_1d(c, GGML_TYPE_F32, ZEROS_MAX);  // TOP_K_MASK's mask row
    if (m->has_moe) {
        b.eye  = ggml_new_tensor_2d(c, GGML_TYPE_F32, m->n_experts, m->n_experts);
        b.tril = ggml_new_tensor_2d(c, GGML_TYPE_F32, Tp, Tp);
    }
    ten *in_sc    = has_sc ? ggml_new_tensor_2d(c, GGML_TYPE_F32, VOCAB, T) : NULL;
    ten *in_sig   = m->is_rdt ? ggml_new_tensor_2d(c, GGML_TYPE_F32, m->loop_dim, n_loops) : NULL;
    ten *in_noise = has_noise ? ggml_new_tensor_2d(c, GGML_TYPE_F32, D, Tp) : NULL;
    ten *inputs[] = { in_ids, in_tf, b.rope_cos, b.rope_sin, b.consts, b.zeros, b.eye, b.tril, in_sc, in_sig, in_noise };
    for (size_t i = 0; i < sizeof inputs / sizeof *inputs; i++) if (inputs[i]) ggml_set_input(inputs[i]);

    // NoiseEmbedding MLP (the sinusoid features come from the host).
    ten *temb = ggml_add(c, ggml_mul_mat(c, m->ne0_w, in_tf), m->ne0_b);
    temb = ggml_add(c, ggml_mul_mat(c, m->ne2_w, ggml_silu(c, temb)), m->ne2_b);
    // Every block's t_proj(t_emb) in one matmul; recurrent blocks reuse theirs
    // on every loop instead of recomputing it.
    b.tbias = ggml_mul_mat(c, m->tproj_all, temb);  // [hidden * n_blocks, 1]

    ten *x = ggml_get_rows(c, m->embed, in_ids);  // [D, T]
    if (in_sc) {
        ten *ev = ggml_cont(c, ggml_transpose(c, ggml_view_2d(c, m->embed, D, VOCAB, m->embed->nb[1], 0)));
        x = ggml_add(c, x, ggml_mul_mat(c, m->self_cond_proj, ggml_mul_mat(c, ev, in_sc)));
    }

    ten *skip = NULL;
    if (m->subpos) {
        x = ggml_reshape_2d(c, ggml_add(c, ggml_reshape_3d(c, x, D, P, Tp), m->subpos), D, T);
        skip = local_residual(&b, x, m->enc_norm, m->enc_conv);
        x = lin(&b, m->cmp, norm(&b, ggml_reshape_2d(c, skip, (int64_t) D * P, Tp), m->cmp_norm), Q_INT8);
    }

    for (int i = 0; i < m->n_pre; i++) x = block(&b, &m->pre[i], x, temb);
    if (m->is_rdt) x = recurrent(&b, x, temb, n_loops, in_noise, in_sig);
    for (int i = 0; i < m->n_post; i++) x = block(&b, &m->post[i], x, temb);

    ten *logits = NULL;
    if (kv != KV_COMMIT) {  // a commit only needs the K/V writes already in the graph
        x = norm(&b, x, m->final_norm);
        if (m->subpos) {
            ten *e = ggml_reshape_2d(c, lin(&b, m->exp, x, Q_INT8), D, T);
            x = ggml_scale(c, ggml_add(c, e, skip), 0.70710678f);  // 2 ** -0.5
            x = local_residual(&b, x, m->dec_norm, m->dec_conv);
            if (m->dec_ffn.up) x = ggml_add(c, x, ffn(&b, &m->dec_ffn, norm(&b, x, m->dec_ffn_norm)));
            x = norm(&b, x, m->out_norm);
        }
        logits = ggml_mul_mat(c, m->head, x);  // [rows, T]
        ggml_set_output(logits);
        ggml_build_forward_expand(b.gf, logits);
    }

    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, b.gf)) die("graph allocation failed");
    *bp = b;
    G.in_ids = in_ids; G.in_tf = in_tf; G.in_sc = in_sc; G.in_sig = in_sig; G.in_noise = in_noise;
    G.logits = logits;
}

// One denoiser call. ids: T byte IDs (T % patch == 0). sc: optional
// self-conditioning probabilities [T][VOCAB]. out: logits [T][embed rows]
// (unused for KV_COMMIT). kv: KV_NONE (full canvas), KV_READ (attend over the
// committed cache), KV_COMMIT (store this canvas at t = 0).
static void forward(model_t *m, const int32_t *ids, int T, float t, int n_loops,
                    const float *sc, const float *noise, float *out, int kv) {
    int D = m->hidden, P = m->patch, Tp = T / P, hd = m->head_dim;
    if (kv != KV_NONE) {
        if (m->is_rdt && n_loops != m->kv_loops) die("KV cache was sized for %d loops", m->kv_loops);
        if (m->kv_len + Tp > m->kv_cap) die("KV cache full");
    }
    int has_sc = sc && m->self_cond_proj, has_noise = m->is_rdt && noise;
    // YEOUBI_NO_GRAPH_REUSE=1 rebuilds every call; CI checks both give
    // byte-identical samples.
    static int no_reuse = -1;
    if (no_reuse < 0) no_reuse = getenv("YEOUBI_NO_GRAPH_REUSE") != NULL;
    // Commits always rebuild: their cache offsets move every time.
    if (!G.ctx || no_reuse || kv == KV_COMMIT || G.T != T || G.loops != n_loops || G.kv != kv ||
        G.kv_len != m->kv_len || G.has_sc != has_sc || G.has_noise != has_noise) {
        build_graph(m, T, n_loops, has_sc, has_noise, kv);
    }
    gb_t b = G.b;
    ten *in_ids = G.in_ids, *in_tf = G.in_tf, *in_sc = G.in_sc, *in_sig = G.in_sig;
    ten *in_noise = G.in_noise, *logits = G.logits;

    // Host-side inputs.
    set_input(in_ids, ids, sizeof(int32_t) * T);
    {
        int half = m->t_dim / 2;
        float *tf = calloc(m->t_dim, sizeof(float));
        for (int j = 0; j < half; j++) {
            float a = (float) (t * m->timestep_scale) * expf(-logf(10000.0f) * j / half);
            tf[j] = sinf(a);
            tf[half + j] = cosf(a);
        }
        set_input(in_tf, tf, sizeof(float) * m->t_dim);
        free(tf);
    }
    {
        // With a cache, positions continue after the committed patches.
        float *cs = malloc(sizeof(float) * 2 * hd * Tp);
        rope_tables(m, kv != KV_NONE ? m->kv_len : 0, Tp, cs, cs + (size_t) hd * Tp);
        set_input(b.rope_cos, cs, sizeof(float) * hd * Tp);
        set_input(b.rope_sin, cs + (size_t) hd * Tp, sizeof(float) * hd * Tp);
        free(cs);
    }
    set_input(b.consts, b.cvals, sizeof(float) * N_CONST);
    if (b.zeros) {
        static const float zeros[ZEROS_MAX];
        set_input(b.zeros, zeros, sizeof zeros);
    }
    if (b.eye) {
        int N = m->n_experts;
        float *e = calloc((size_t) N * N + (size_t) Tp * Tp, sizeof(float));
        for (int i = 0; i < N; i++) e[(size_t) i * N + i] = 1.0f;
        float *tl = e + (size_t) N * N;  // tril[t][s] = 1 for s < t (ne0 = s)
        for (int tt = 0; tt < Tp; tt++) for (int s = 0; s < tt; s++) tl[(size_t) tt * Tp + s] = 1.0f;
        set_input(b.eye, e, sizeof(float) * N * N);
        set_input(b.tril, tl, sizeof(float) * Tp * Tp);
        free(e);
    }
    if (in_sc) set_input(in_sc, sc, sizeof(float) * VOCAB * T);
    if (in_noise) set_input(in_noise, noise, sizeof(float) * D * Tp);
    if (in_sig) {
        float *sig = malloc(sizeof(float) * m->loop_dim * n_loops);
        int half = m->loop_dim / 2;
        for (int i = 0; i < n_loops; i++) {
            float *row = sig + (size_t) i * m->loop_dim;
            if (m->loop_embed) {
                int rows = m->n_loop_embed / m->loop_dim;
                memcpy(row, m->loop_embed + (size_t) clampi(i, rows) * m->loop_dim, sizeof(float) * m->loop_dim);
            } else {
                // Legacy sinusoid of i / (max_loop_iters - 1), unclamped like PyTorch.
                float frac = (float) i / (float) (m->max_loops > 1 ? m->max_loops - 1 : 1);
                for (int j = 0; j < half; j++) {
                    float a = frac * expf(-logf(10000.0f) * (float) j / half);
                    row[j] = sinf(a);
                    row[half + j] = cosf(a);
                }
            }
        }
        set_input(in_sig, sig, sizeof(float) * m->loop_dim * n_loops);
        free(sig);
    }

    if (ggml_backend_sched_graph_compute(m->sched, b.gf) != GGML_STATUS_SUCCESS) die("graph compute failed");
    if (logits) ggml_backend_tensor_get(logits, out, 0, ggml_nbytes(logits));
    if (kv == KV_COMMIT) m->kv_len += Tp;
}

// ---------------------------------------------------------------------------
// Samplers (yeoubi/sample.py)
// ---------------------------------------------------------------------------

static uint64_t rng_state;

static double rnd(void) {  // splitmix64 -> [0, 1)
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (double) ((z ^ (z >> 31)) >> 11) / 9007199254740992.0;
}

static double gauss(void) {
    double u = rnd();
    return sqrt(-2.0 * log(u > 0 ? u : 1e-300)) * cos(6.283185307179586 * rnd());
}

static double mask_prob(double t) { return 1.0 - cos(t * 1.5707963267948966); }

typedef struct { float p; int i; } pi_t;

static int by_p_desc(const void *a, const void *b) {
    float x = ((const pi_t *) a)->p, y = ((const pi_t *) b)->p;
    return (x < y) - (x > y);
}

static int by_p_asc(const void *a, const void *b) { return by_p_desc(b, a); }

static void softmax(const float *l, float temp, float *p) {
    float mx = -FLT_MAX, s = 0;
    for (int v = 0; v < VOCAB; v++) mx = fmaxf(mx, l[v]);
    for (int v = 0; v < VOCAB; v++) s += p[v] = expf((l[v] - mx) / temp);
    for (int v = 0; v < VOCAB; v++) p[v] /= s;
}

// Temperature, nucleus filter (keep the first threshold-crossing ID), draw.
static int draw(const float *l, float temp, float top_p) {
    float p[VOCAB];
    pi_t s[VOCAB];
    softmax(l, fmaxf(temp, 1e-8f), p);
    for (int v = 0; v < VOCAB; v++) s[v] = (pi_t) { p[v], v };
    int n = VOCAB;
    if (top_p < 1.0f) {
        qsort(s, VOCAB, sizeof *s, by_p_desc);
        float cum = 0;
        for (n = 0; n < VOCAB && cum < top_p; n++) cum += s[n].p;
    }
    float tot = 0;
    for (int j = 0; j < n; j++) tot += s[j].p;
    double r = rnd() * tot;
    for (int j = 0; j < n; j++) if ((r -= s[j].p) < 0) return s[j].i;
    return s[n - 1].i;
}

typedef struct {
    const char *model, *device, *prompt, *prompt_file, *dump;
    int length, steps, loops, threads, verbose, remask, block, auto_length;  // remask: 0 random, 1 confidence, 2 remdm
    float temp, top_p, eta, revise, revise_t_max, dump_t;
    uint64_t seed;
} opts_t;

static int select_loops(model_t *m, const opts_t *o, double t, int T) {
    if (o->loops > 0) return o->loops;
    if (!m->adaptive_loops) return m->max_loops;
    double noise = fmin(fmax(t, 0.0), 1.0);
    int nl = m->min_loops + (int) ceil(noise * (m->max_loops - m->min_loops) - 1e-8);
    int ll = m->min_loops;
    if (m->context_adaptive && T > m->context_threshold) ll += (int) ceil(log2((double) T / m->context_threshold));
    nl = nl > ll ? nl : ll;
    return nl < m->max_loops ? nl : m->max_loops;
}

static float *make_noise(model_t *m, int T) {
    if (!m->is_rdt || m->state_std <= 0) return NULL;
    size_t n = (size_t) m->hidden * (T / m->patch);
    float *z = malloc(sizeof(float) * n);
    for (size_t i = 0; i < n; i++) z[i] = (float) gauss();
    return z;
}

// The denoising schedule over one canvas, in place. Positions < frozen are
// prompt. fixed_loops > 0 pins the recurrent depth (required with a cache).
static void denoise(model_t *m, const opts_t *o, int32_t *ids, int T, int frozen, int kv,
                    int fixed_loops, float *logits) {
    int rows = (int) m->head->ne[1];
    char *masked = malloc(T);
    float *sc = NULL, p[VOCAB];
    for (int s = 0; s < o->steps; s++) {
        // Uniform in mask probability (CosineSchedule.inverse of linspace(1, 0)).
        double tc = acos((double) s / o->steps) / 1.5707963267948966;
        double tn = acos((double) (s + 1) / o->steps) / 1.5707963267948966;
        int nl = fixed_loops > 0 ? fixed_loops : select_loops(m, o, tc, T);
        float *z = make_noise(m, T);
        int64_t t0 = ggml_time_us();
        forward(m, ids, T, (float) tc, nl, sc, z, logits, kv);
        double ms = (ggml_time_us() - t0) / 1000.0;
        free(z);
        if (m->self_cond_proj) {  // self_cond_vector: next step sees this step's beliefs
            if (!sc) sc = malloc(sizeof(float) * VOCAB * T);
            for (int i = 0; i < T; i++) softmax(logits + (size_t) i * rows, 1.0f, sc + (size_t) i * VOCAB);
        }

        for (int i = 0; i < T; i++) {
            masked[i] = ids[i] == MASK_ID;
            if (masked[i]) ids[i] = draw(logits + (size_t) i * rows, o->temp, o->top_p);
        }
        if (o->revise > 0 && tc <= o->revise_t_max) {  // _revise_step
            for (int i = frozen; i < T; i++) {
                if (masked[i]) continue;
                softmax(logits + (size_t) i * rows, 1.0f, p);
                int best = 0;
                for (int v = 1; v < VOCAB; v++) if (p[v] > p[best]) best = v;
                if (best != ids[i] && p[best] > o->revise) ids[i] = best;
            }
        }
        if (s + 1 < o->steps) {  // _remask_step
            double pc = fmax(mask_prob(tc), 1e-6), pn = mask_prob(tn), committed = 1.0 - pc;
            double sigma = o->remask == 2 ? fmin(pn / fmax(committed, 1e-6), o->eta) : 0.0;
            double ratio = (pn - sigma * committed) / pc;
            if (o->remask == 1) {  // least confident of the bytes written this step
                pi_t *conf = malloc(sizeof(pi_t) * T);
                int n = 0;
                for (int i = frozen; i < T; i++) {
                    if (!masked[i]) continue;
                    softmax(logits + (size_t) i * rows, 1.0f, p);
                    conf[n++] = (pi_t) { p[ids[i] < VOCAB ? ids[i] : VOCAB - 1], i };
                }
                qsort(conf, n, sizeof *conf, by_p_asc);
                int k = (int) nearbyint(n * ratio);
                for (int j = 0; j < k && j < n; j++) ids[conf[j].i] = MASK_ID;
                free(conf);
            } else {
                for (int i = frozen; i < T; i++) {
                    double r = rnd();
                    if (masked[i] ? r < ratio : r < sigma) ids[i] = MASK_ID;
                }
            }
        }
        if (o->verbose) {
            int n = 0;
            for (int i = 0; i < T; i++) n += ids[i] == MASK_ID;
            fprintf(stderr, "step %d/%d t=%.3f->%.3f loops=%d masked=%d forward=%.1f ms\n",
                    s + 1, o->steps, tc, tn, nl, n, ms);
        }
    }
    free(masked);
    free(sc);
}

static void emit(const int32_t *ids, int n) {
    // Byte IDs only, like decode_byte_ids; the terminal does UTF-8 decoding.
    for (int i = 0; i < n; i++) if (ids[i] < 256) fputc(ids[i], stdout);
    fflush(stdout);
}

static void write_logits(const char *path, const float *logits, int T, int rows) {
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write %s", path);
    fwrite(logits, sizeof(float), (size_t) rows * T, f);
    fclose(f);
    fprintf(stderr, "wrote %d x %d logits to %s\n", T, rows, path);
}

// Full-canvas sampler (denoise). --dump-logits: one forward, raw float32.
static int run_canvas(model_t *m, const opts_t *o, const unsigned char *prompt, int plen) {
    int P = m->patch, rows = (int) m->head->ne[1];
    int T = o->length > plen ? o->length : plen + 1;
    T = (T + P - 1) / P * P;  // the canvas fills whole patches; the extra bytes are generated too
    int32_t *ids  = malloc(sizeof(int32_t) * T);
    float *logits = malloc(sizeof(float) * rows * T);
    for (int i = 0; i < T; i++) ids[i] = i < plen ? prompt[i] : MASK_ID;
    if (o->dump) {
        float *z = make_noise(m, T);
        forward(m, ids, T, o->dump_t, select_loops(m, o, o->dump_t, T), NULL, z, logits, KV_NONE);
        free(z);
        write_logits(o->dump, logits, T, rows);
    } else {
        denoise(m, o, ids, T, plen, KV_NONE, 0, logits);
        emit(ids, T);
        fputc('\n', stdout);
    }
    free(ids);
    free(logits);
    return 0;
}

// BlockDiffusionSampler: blocks left to right, each denoised against the
// committed cache, then committed at t = 0. -n is the number of bytes to
// generate (the cap with --auto-length); output streams block by block.
static int run_block(model_t *m, const opts_t *o, const unsigned char *prompt, int plen) {
    int P = m->patch, B = o->block, rows = (int) m->head->ne[1];
    if (B % P) die("--block must be a multiple of the byte patch size (%d)", P);
    // A cached forward uses one fixed depth (rdt.py: max_loop_iters unless given).
    int loops = o->loops > 0 ? o->loops : m->max_loops;
    // Commit only whole patches of the prompt; the 0..P-1 leftover bytes ride
    // as frozen literals at the head of the first block.
    int commit_len = plen / P * P, carry = plen - commit_len;
    int total = commit_len + carry + o->length + B;
    kv_alloc(m, loops, total / P + 2);

    int32_t *ids  = malloc(sizeof(int32_t) * B);
    float *logits = malloc(sizeof(float) * rows * B);
    // Prompt in block-sized chunks: block-causal training never saw longer spans.
    for (int start = 0; start < commit_len; start += B) {
        int n = commit_len - start < B ? commit_len - start : B;
        for (int i = 0; i < n; i++) ids[i] = prompt[start + i];
        float *z = make_noise(m, n);
        forward(m, ids, n, 0.0f, loops, NULL, z, NULL, KV_COMMIT);
        free(z);
    }

    if (!o->dump && plen) {  // echo the prompt, as the full-canvas output includes it
        fwrite(prompt, 1, plen, stdout);
        fflush(stdout);
    }
    int generated = 0;
    for (int first = 1; generated < o->length; first = 0) {
        int nc = first ? carry : 0;
        int nn = B - nc < o->length - generated ? B - nc : o->length - generated;
        int T = (nn + nc + P - 1) / P * P;  // whole patches; the few extra bytes are generated too
        for (int i = 0; i < T; i++) ids[i] = i < nc ? prompt[commit_len + i] : MASK_ID;
        if (o->dump) {
            float *z = make_noise(m, T);
            forward(m, ids, T, o->dump_t, loops, NULL, z, logits, KV_READ);
            free(z);
            write_logits(o->dump, logits, T, rows);
            break;
        }
        int64_t t0 = ggml_time_us();
        denoise(m, o, ids, T, nc, KV_READ, loops, logits);
        float *z = make_noise(m, T);
        forward(m, ids, T, 0.0f, loops, NULL, z, NULL, KV_COMMIT);
        free(z);
        int n_out = T - nc, stop = 0;
        if (o->auto_length) {
            for (int i = nc; i < T; i++) if (ids[i] == EOS_ID) { n_out = i - nc; stop = 1; break; }
        }
        emit(ids + nc, n_out);
        generated += T - nc;
        if (o->verbose) {
            double s = (ggml_time_us() - t0) / 1e6;
            fprintf(stderr, "\nblock: %d bytes in %.2f s (%.0f bytes/s)\n", T - nc, s, (T - nc) / s);
        }
        if (stop) break;
    }
    if (!o->dump) fputc('\n', stdout);
    free(ids);
    free(logits);
    return 0;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void usage(void) {
    fprintf(stderr,
        "usage: yeoubi -m model.gguf [options]\n"
        "  --list-devices        print ggml devices and exit\n"
        "  -d, --device NAME     device from --list-devices (default: best available)\n"
        "  -t, --threads N       CPU threads (default 4)\n"
        "  -p, --prompt TEXT     frozen prompt prefix\n"
        "  -f, --prompt-file F   read the prompt bytes from a file (use for non-ASCII on Windows)\n"
        "  -n, --length N        canvas bytes incl. prompt (default 256); with --block: bytes to generate\n"
        "  --block N             block diffusion with a KV cache, N bytes per block (multiple of the patch size)\n"
        "  --auto-length         with --block: stop at the first EOS\n"
        "  --steps N             denoising steps (per block with --block, default 20)\n"
        "  --temp F              temperature (default 0.9)\n"
        "  --top-p F             nucleus threshold (default 0.95)\n"
        "  --seed N              RNG seed (default 42)\n"
        "  --loops N             recurrent loop override (RDT)\n"
        "  --remask MODE         random | confidence | remdm (default random)\n"
        "  --eta F               remdm cap (default 0.04)\n"
        "  --revise F            overwrite visible bytes above this confidence (default off)\n"
        "  --revise-t-max F      only revise when t <= F (default 0.5)\n"
        "  --act a4|a8|none      activation quant: trained A4 stage (default), A8 stage, off\n"
        "  --weight-type T       q4 | tq2 | tq1 | f16: in-memory ternary format (default: fastest for the device)\n"
        "  --dump-logits F       one forward at --t, write float32 logits, exit\n"
        "  --t F                 noise level for --dump-logits (default 1.0)\n"
        "  -v, --verbose         per-step progress and timing on stderr\n");
    exit(1);
}

static void set_threads(ggml_backend_t be, int n) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(be));
    ggml_backend_set_n_threads_t fn =
        (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) fn(be, n);
}

int main(int argc, char **argv) {
    opts_t o = { 0 };
    o.length = 256; o.steps = 20; o.threads = 4; o.temp = 0.9f; o.top_p = 0.95f; o.seed = 42;
    o.eta = 0.04f; o.revise_t_max = 0.5f; o.dump_t = 1.0f;
    int act = ACT_A4, list = 0, weight_type = -1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
#define ARG(s, l) ((!strcmp(a, s) || !strcmp(a, l)) && more)
#define OPT(l) (!strcmp(a, l) && more)
        if      (ARG("-m", "--model"))       o.model = argv[++i];
        else if (ARG("-d", "--device"))      o.device = argv[++i];
        else if (ARG("-t", "--threads"))     o.threads = atoi(argv[++i]);
        else if (ARG("-p", "--prompt"))      o.prompt = argv[++i];
        else if (ARG("-f", "--prompt-file")) o.prompt_file = argv[++i];
        else if (ARG("-n", "--length"))      o.length = atoi(argv[++i]);
        else if (OPT("--block"))             o.block = atoi(argv[++i]);
        else if (OPT("--steps"))             o.steps = atoi(argv[++i]);
        else if (OPT("--temp"))              o.temp = (float) atof(argv[++i]);
        else if (OPT("--top-p"))             o.top_p = (float) atof(argv[++i]);
        else if (OPT("--seed"))              o.seed = strtoull(argv[++i], NULL, 10);
        else if (OPT("--loops"))             o.loops = atoi(argv[++i]);
        else if (OPT("--eta"))               o.eta = (float) atof(argv[++i]);
        else if (OPT("--revise"))            o.revise = (float) atof(argv[++i]);
        else if (OPT("--revise-t-max"))      o.revise_t_max = (float) atof(argv[++i]);
        else if (OPT("--dump-logits"))       o.dump = argv[++i];
        else if (OPT("--t"))                 o.dump_t = (float) atof(argv[++i]);
        else if (OPT("--act")) {
            const char *v = argv[++i];
            act = !strcmp(v, "a4") ? ACT_A4 : !strcmp(v, "a8") ? ACT_A8 : !strcmp(v, "none") ? ACT_NONE : -1;
            if (act < 0) usage();
        }
        else if (OPT("--weight-type")) {
            const char *v = argv[++i];
            weight_type = !strcmp(v, "q4") ? GGML_TYPE_Q4_0 : !strcmp(v, "tq2") ? GGML_TYPE_TQ2_0
                        : !strcmp(v, "tq1") ? GGML_TYPE_TQ1_0 : !strcmp(v, "f16") ? GGML_TYPE_F16 : -2;
            if (weight_type == -2) usage();
        }
        else if (OPT("--remask")) {
            const char *r = argv[++i];
            o.remask = !strcmp(r, "confidence") ? 1 : !strcmp(r, "remdm") ? 2 : !strcmp(r, "random") ? 0 : -1;
            if (o.remask < 0) usage();
        }
        else if (!strcmp(a, "--auto-length")) o.auto_length = 1;
        else if (!strcmp(a, "--list-devices")) list = 1;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) o.verbose = 1;
        else usage();
#undef ARG
#undef OPT
    }
    if (o.steps < 1 || o.length < 1 || o.loops > MAX_LOOPS || o.block < 0) usage();

    ggml_time_init();
    ggml_backend_load_all();  // dynamic backends (GGML_BACKEND_DL); no-op for static builds
    if (list) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            printf("%-16s %s\n", ggml_backend_dev_name(d), ggml_backend_dev_description(d));
        }
        return 0;
    }
    if (!o.model) usage();

    model_t m = { 0 };
    m.act = act;
    if (o.device) {
        ggml_backend_dev_t d = ggml_backend_dev_by_name(o.device);
        if (!d) die("no device %s (see --list-devices)", o.device);
        m.backend = ggml_backend_dev_init(d, NULL);
    } else {
        m.backend = ggml_backend_init_best();
    }
    if (!m.backend) die("failed to initialise a backend");
    int on_cpu = ggml_backend_dev_type(ggml_backend_get_device(m.backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    m.cpu = on_cpu ? m.backend : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!m.cpu) die("failed to initialise the CPU backend");
    set_threads(m.cpu, o.threads);
    fprintf(stderr, "device: %s\n", ggml_backend_dev_name(ggml_backend_get_device(m.backend)));

    model_load(&m, o.model, weight_type);
    if (m.is_rdt && m.max_loops > MAX_LOOPS) die("max_loop_iters > %d", MAX_LOOPS);
    // Unsupported ops on the main device fall back to the CPU (always last).
    ggml_backend_t backends[2] = { m.backend, m.cpu };
    int visits = m.n_pre + m.n_post + (m.is_rdt ? m.n_rec * (o.loops > m.max_loops ? o.loops : m.max_loops) : 0);
    m.sched = ggml_backend_sched_new(backends, NULL, on_cpu ? 1 : 2, (size_t) 320 * (visits + 4) + 4096, false, false);

    unsigned char *prompt = NULL;
    int plen = 0;
    if (o.prompt_file) {
        FILE *f = fopen(o.prompt_file, "rb");
        if (!f) die("cannot open %s", o.prompt_file);
        fseek(f, 0, SEEK_END);
        plen = (int) ftell(f);
        fseek(f, 0, SEEK_SET);
        prompt = malloc(plen + 1);
        plen = (int) fread(prompt, 1, plen, f);
        fclose(f);
    } else if (o.prompt) {
        plen = (int) strlen(o.prompt);
        prompt = (unsigned char *) o.prompt;
    }

#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);  // raw UTF-8 bytes, no CRLF rewriting
#endif
    rng_state = o.seed;
    int rc = o.block ? run_block(&m, &o, prompt, plen) : run_canvas(&m, &o, prompt, plen);

    if (G.ctx) ggml_free(G.ctx);
    ggml_backend_sched_free(m.sched);
    if (m.kvbuf) ggml_backend_buffer_free(m.kvbuf);
    if (m.kvctx) ggml_free(m.kvctx);
    ggml_backend_buffer_free(m.wbuf);
    ggml_free(m.wctx);
    if (!on_cpu) ggml_backend_free(m.cpu);
    ggml_backend_free(m.backend);
    return rc;
}
