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
 * Input is a GGUF from `python export.py --format gguf` (yeoubi/gguf_export.py).
 * The forward pass mirrors yeoubi/model.py + yeoubi/rdt.py at eval time; the
 * sampler mirrors yeoubi/sample.py:denoise(). Check parity with
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

#define MASK_ID    258
#define VOCAB      258   // predictable IDs: bytes 0-255, PAD, EOS (byte_codec.py)
#define GRAPH_SIZE 32768 // nodes; ~75 per block visit, so ~400 visits of headroom
#define MAX_LOOPS  256

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
enum { A_NONE, A_INT4, A_TOPK };

typedef struct { ten *up, *gate, *down, *sub; } ffn_t;

typedef struct {
    ten *attn_norm, *ffn_norm, *t_proj;
    ten *wq, *wk, *wv, *wo, *q_norm, *k_norm, *attn_sub;
    ffn_t ffn;
} block_t;

typedef struct {
    // config
    int    n_heads, head_dim, hidden, patch, t_dim, loop_dim, max_loops, min_loops;
    int    int4_absmean, adaptive_loops, context_adaptive, context_threshold, rope_type;
    double topk_ratio, rope_theta, rope_factor, timestep_scale, state_std;
    // runtime
    int act_a4;
    ggml_backend_t backend, cpu;
    ggml_backend_sched_t sched;
    struct ggml_context *wctx;
    ggml_backend_buffer_t wbuf;
    // weights (NULL = absent in this checkpoint)
    ten *embed, *ne0_w, *ne0_b, *ne2_w, *ne2_b, *final_norm, *self_cond_proj;
    ten *subpos, *enc_norm, *enc_conv, *cmp_norm, *cmp, *exp, *dec_norm, *dec_conv;
    ten *dec_ffn_norm, *out_norm;
    ffn_t dec_ffn;
    block_t *pre, *rec, *post;   // pre = prelude + pre_recurrent (or blocks), post = post_recurrent + coda
    int n_pre, n_rec, n_post, is_rdt;
    ten *inj_A, *inj_B, *lora_down, *lora_up, *act_w, *loop_proj, *loop_proj2, *loop_proj2_b, *loop_norm;
    ten *loop_norms[MAX_LOOPS];
    int n_loop_norms;
    float *alpha, *iter_gate, *loop_embed;  // tiny host copies used as graph constants
    int n_alpha, n_gate, n_loop_embed;
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
    f->up   = getf(m, 0, "%s.up_proj.weight", p);
    if (!f->up) return;
    f->gate = getf(m, 1, "%s.gate_proj.weight", p);
    f->down = getf(m, 1, "%s.down_proj.weight", p);
    f->sub  = getf(m, 0, "%s.sub_norm.weight", p);
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
        b->t_proj    = getf(m, 1, "%s.t_proj.weight", p);
        b->wq        = getf(m, 1, "%s.attn.q_proj.weight", p);
        b->wk        = getf(m, 1, "%s.attn.k_proj.weight", p);
        b->wv        = getf(m, 1, "%s.attn.v_proj.weight", p);
        b->wo        = getf(m, 1, "%s.attn.o_proj.weight", p);
        b->q_norm    = getf(m, 0, "%s.attn.q_norm.weight", p);
        b->k_norm    = getf(m, 0, "%s.attn.k_norm.weight", p);
        b->attn_sub  = getf(m, 0, "%s.attn.sub_norm.weight", p);
        snprintf(p, sizeof p, "%s.%d.ffn", prefix, i);
        load_ffn(m, &b->ffn, p);
        if (!b->ffn.up) die("%s: missing FFN (MoE checkpoints are not supported)", p);
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

static void model_load(model_t *m, const char *path) {
    struct ggml_context *hctx = NULL;
    struct gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &hctx };
    struct gguf_context *g = gguf_init_from_file(path, gp);
    if (!g) die("cannot read GGUF %s", path);
    const char *type = cfg_str(g, "model_type", NULL);
    if (!type) die("%s is not a Yeoubi export (run: python export.py --format gguf)", path);
    if (cfg_num(g, "use_moe", 0)) die("MoE checkpoints are not supported yet");
    if (cfg_num(g, "vocab_size", VOCAB) != VOCAB) die("not a byte-level checkpoint");

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
    const char *rope = cfg_str(g, "rope_scaling_type", "none");
    // Matches RotaryEmbedding._scaled: scaling only applies with factor > 1.
    m->rope_type = m->rope_factor <= 1.0       ? ROPE_NONE
                 : strcmp(rope, "linear") == 0 ? ROPE_LINEAR
                 : strcmp(rope, "yarn") == 0   ? ROPE_YARN : ROPE_NONE;

    // Weights: mirror every GGUF tensor into one buffer on the chosen device.
    int64_t nt = gguf_get_n_tensors(g);
    struct ggml_init_params ip = { (size_t) nt * ggml_tensor_overhead(), NULL, true };
    m->wctx = ggml_init(ip);
    for (int64_t i = 0; i < nt; i++) {
        const char *name = gguf_get_tensor_name(g, i);
        ggml_set_name(ggml_dup_tensor(m->wctx, ggml_get_tensor(hctx, name)), name);
    }
    m->wbuf = ggml_backend_alloc_ctx_tensors_from_buft(m->wctx, ggml_backend_get_default_buffer_type(m->backend));
    if (!m->wbuf) die("out of device memory for weights");
    for (int64_t i = 0; i < nt; i++) {
        const char *name = gguf_get_tensor_name(g, i);
        ten *src = ggml_get_tensor(hctx, name);
        ggml_backend_tensor_set(get(m, name), src->data, 0, ggml_nbytes(src));
    }
    m->alpha      = host_copy(hctx, "recurrent.alpha", &m->n_alpha);
    m->iter_gate  = host_copy(hctx, "recurrent.lora.iter_gate", &m->n_gate);
    m->loop_embed = host_copy(hctx, "recurrent.loop_embed.weight", &m->n_loop_embed);
    ggml_free(hctx);
    gguf_free(g);

    m->embed  = need(m, "embed.weight");
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
        m->act_w        = get(m, "recurrent.act.weight_proj.weight");
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

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

typedef struct {
    struct ggml_context *ctx;
    model_t *m;
    ten *pos, *freq_factors, *one;
} gb_t;

static ten *norm(gb_t *b, ten *x, ten *w) {
    if (!w) return x;  // nn.Identity in the PyTorch model
    return ggml_mul(b->ctx, ggml_rms_norm(b->ctx, x, 1e-6f), w);
}

// Per-row k-th largest value (1-based) of a contiguous [n, rows] tensor -> [1, rows].
static ten *kth_largest(gb_t *b, ten *a, int k) {
    struct ggml_context *c = b->ctx;
    int64_t n = a->ne[0], rows = a->ne[1];
    ten *idx = ggml_argsort(c, a, GGML_SORT_ORDER_DESC);
    idx = ggml_cont(c, ggml_view_2d(c, idx, 1, rows, idx->nb[1], (size_t) (k - 1) * idx->nb[0]));
    // Treat each element as a 1-wide row so get_rows gathers one value per row.
    ten *v = ggml_get_rows(c, ggml_reshape_3d(c, a, 1, n, rows), idx);
    return ggml_reshape_2d(c, v, 1, rows);
}

// HybridQuantizer "int4": per-token scale, round, clamp +/-7, dequantize.
static ten *q_int4(gb_t *b, ten *x) {
    struct ggml_context *c = b->ctx;
    ten *a = ggml_abs(c, x), *step;
    if (b->m->int4_absmean) step = ggml_scale(c, ggml_mean(c, a), 1.0f / sqrtf(7.0f));  // beta / sqrt(7)
    else                    step = ggml_scale(c, kth_largest(b, a, 1), 1.0f / 7.0f);  // amax / 7
    step = ggml_clamp(c, step, 1e-8f, FLT_MAX);
    ten *q = ggml_clamp(c, ggml_round(c, ggml_div(c, x, step)), -7.0f, 7.0f);
    return ggml_mul(c, q, step);
}

// HybridQuantizer "topk_int8": keep the int(n * ratio) largest |x| per token.
// ponytail: the INT8 rounding after it (and the plain "int8" mode) is skipped:
// 127 levels per token is below ggml's own Q8 activation blocking noise.
static ten *topk(gb_t *b, ten *x) {
    struct ggml_context *c = b->ctx;
    int64_t n = x->ne[0];
    int k = (int) (n * b->m->topk_ratio);
    if (k < 1) k = 1;
    if (k >= n) return x;
    ten *a   = ggml_abs(c, x);
    ten *thr = ggml_scale(c, kth_largest(b, a, k), 1.0f - 1e-6f);  // keep the k-th itself
    return ggml_mul(c, x, ggml_step(c, ggml_sub(c, a, thr)));
}

static ten *lin(gb_t *b, ten *w, ten *x, int mode) {
    if (b->m->act_a4) {
        if (mode == A_INT4) x = q_int4(b, x);
        if (mode == A_TOPK) x = topk(b, x);
    }
    return ggml_mul_mat(b->ctx, w, x);
}

static ten *ffn(gb_t *b, ffn_t *f, ten *h) {
    struct ggml_context *c = b->ctx;
    if (b->m->act_a4) h = q_int4(b, h);  // up and gate share one quantized input
    ten *mid = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, f->gate, h)), ggml_mul_mat(c, f->up, h));
    return lin(b, f->down, norm(b, mid, f->sub), A_TOPK);
}

static ten *rope(gb_t *b, ten *x) {
    model_t *m = b->m;
    float freq_scale = m->rope_type == ROPE_LINEAR ? (float) (1.0 / m->rope_factor) : 1.0f;
    // YaRN attention temperature, folded into cos/sin as in RotaryEmbedding.
    float mscale = m->rope_type == ROPE_YARN ? (float) (0.1 * log(m->rope_factor) + 1.0) : 1.0f;
    return ggml_rope_ext(b->ctx, x, b->pos, b->freq_factors, m->head_dim, GGML_ROPE_TYPE_NEOX, 0,
                         (float) m->rope_theta, freq_scale, 0.0f, mscale, 0.0f, 0.0f);
}

static ten *attention(gb_t *b, block_t *L, ten *h) {
    struct ggml_context *c = b->ctx;
    int hd = b->m->head_dim, H = b->m->n_heads;
    int64_t T = h->ne[1];
    if (b->m->act_a4) h = q_int4(b, h);  // q, k, v share one quantized input
    ten *q = ggml_reshape_3d(c, ggml_mul_mat(c, L->wq, h), hd, H, T);
    ten *k = ggml_reshape_3d(c, ggml_mul_mat(c, L->wk, h), hd, H, T);
    ten *v = ggml_reshape_3d(c, ggml_mul_mat(c, L->wv, h), hd, H, T);
    q = rope(b, norm(b, q, L->q_norm));
    k = rope(b, norm(b, k, L->k_norm));
    q = ggml_permute(c, q, 0, 2, 1, 3);                 // [hd, T, H]
    k = ggml_permute(c, k, 0, 2, 1, 3);
    v = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));   // [T, hd, H]
    // Bidirectional, unpadded canvas: no mask at all.
    // ponytail: plain softmax attention (T^2 per head); swap in
    // ggml_flash_attn_ext when canvases get long enough for that to matter.
    ten *kq = ggml_soft_max_ext(c, ggml_mul_mat(c, k, q), NULL, 1.0f / sqrtf((float) hd), 0.0f);
    ten *o  = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, v, kq), 0, 2, 1, 3));  // [hd, H, T]
    o = ggml_reshape_2d(c, o, (int64_t) hd * H, T);
    return lin(b, L->wo, norm(b, o, L->attn_sub), A_TOPK);
}

static ten *block(gb_t *b, block_t *L, ten *x, ten *temb) {
    struct ggml_context *c = b->ctx;
    ten *h = ggml_add(c, norm(b, x, L->attn_norm), ggml_mul_mat(c, L->t_proj, temb));
    x = ggml_add(c, x, attention(b, L, h));
    return ggml_add(c, x, ffn(b, &L->ffn, norm(b, x, L->ffn_norm)));
}

// BytePatchCodec._local_residual: x + depthwise_conv(norm(x)), zero padded,
// written as K shifted multiply-adds so every backend runs it natively.
static ten *local_residual(gb_t *b, ten *x, ten *nw, ten *w) {
    struct ggml_context *c = b->ctx;
    ten *l = norm(b, x, nw);
    int64_t D = l->ne[0], T = l->ne[1], K = w->ne[1];
    ten *p = ggml_pad_ext(c, l, 0, 0, (int) (K / 2), (int) (K / 2), 0, 0, 0, 0);
    ten *acc = NULL;
    for (int64_t k = 0; k < K; k++) {
        ten *term = ggml_mul(c, ggml_view_2d(c, p, D, T, p->nb[1], (size_t) k * p->nb[1]),
                             ggml_view_1d(c, w, D, (size_t) k * w->nb[1]));
        acc = acc ? ggml_add(c, acc, term) : term;
    }
    return ggml_add(c, x, acc);
}

static int clampi(int i, int n) { return i < n ? i : n - 1; }

// Inputs a given model never reads (b.one without soft ACT, ...) are pruned
// from the graph and get no buffer; writing to them would assert.
static void set_input(ten *t, const void *data, size_t size) {
    if (t && t->buffer) ggml_backend_tensor_set(t, data, 0, size);
}

static ten *recurrent(gb_t *b, ten *x, ten *temb, int n_loops, ten *noise, ten *sig) {
    struct ggml_context *c = b->ctx;
    model_t *m = b->m;
    // Loop invariants: B(e) and the timestep projections never change per loop.
    ten *Be = ggml_mul_mat(c, m->inj_B, x);
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
        ten *lora = lin(b, m->lora_up, ggml_mul_mat(c, m->lora_down, bo), A_TOPK);
        bo = ggml_add(c, bo, ggml_scale(c, lora, m->iter_gate[clampi(i, m->n_gate)]));
        bo = ggml_scale(c, bo, m->alpha[clampi(i, m->n_alpha)]);
        h = ggml_add(c, ggml_add(c, ggml_mul(c, h, m->inj_A), Be), bo);
        hs[i] = h;
        if (m->act_w) ws[i] = ggml_sigmoid(c, ggml_mul_mat(c, m->act_w, h));  // [1, T]
    }
    if (!m->act_w) return h;

    // Soft ACT blend (BitRecurrentBlock.forward): normalise weights across
    // loops, sequential lerp, divide out the achieved total weight.
    ten *sum = ws[0];
    for (int i = 1; i < n_loops; i++) sum = ggml_add(c, sum, ws[i]);
    sum = ggml_add(c, sum, ggml_scale(c, b->one, 1e-6f));
    ten *out = NULL, *keep = NULL;
    for (int i = 0; i < n_loops; i++) {
        ten *wn = ggml_div(c, ws[i], sum);
        out = out ? ggml_add(c, out, ggml_mul(c, ggml_sub(c, hs[i], out), wn)) : ggml_mul(c, hs[i], wn);
        ten *rest = ggml_add(c, ggml_scale(c, wn, -1.0f), b->one);  // 1 - wn
        keep = keep ? ggml_mul(c, keep, rest) : rest;
    }
    ten *total = ggml_clamp(c, ggml_add(c, ggml_scale(c, keep, -1.0f), b->one), 1e-6f, FLT_MAX);
    return ggml_div(c, out, total);
}

// One denoiser call. ids: T byte IDs (T % patch == 0). sc: optional
// self-conditioning probabilities [T][VOCAB]. out: logits [T][embed rows].
static void forward(model_t *m, const int32_t *ids, int T, float t, int n_loops,
                    const float *sc, const float *noise, float *out) {
    size_t mem = ggml_tensor_overhead() * GRAPH_SIZE * 2 + ggml_graph_overhead_custom(GRAPH_SIZE, false);
    struct ggml_init_params ip = { mem, NULL, true };
    struct ggml_context *c = ggml_init(ip);
    gb_t b = { c, m, NULL, NULL, NULL };
    int D = m->hidden, P = m->patch, Tp = T / P, hd = m->head_dim;

    ten *in_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ten *in_tf  = ggml_new_tensor_2d(c, GGML_TYPE_F32, m->t_dim, 1);
    b.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, Tp);
    b.one = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1);
    if (m->rope_type == ROPE_YARN) b.freq_factors = ggml_new_tensor_1d(c, GGML_TYPE_F32, hd / 2);
    ten *in_sc    = sc && m->self_cond_proj ? ggml_new_tensor_2d(c, GGML_TYPE_F32, VOCAB, T) : NULL;
    ten *in_sig   = m->is_rdt ? ggml_new_tensor_2d(c, GGML_TYPE_F32, m->loop_dim, n_loops) : NULL;
    ten *in_noise = m->is_rdt && noise ? ggml_new_tensor_2d(c, GGML_TYPE_F32, D, Tp) : NULL;
    ten *inputs[] = { in_ids, in_tf, b.pos, b.one, b.freq_factors, in_sc, in_sig, in_noise };
    for (size_t i = 0; i < sizeof inputs / sizeof *inputs; i++) if (inputs[i]) ggml_set_input(inputs[i]);

    // NoiseEmbedding MLP (the sinusoid features come from the host).
    ten *temb = ggml_add(c, ggml_mul_mat(c, m->ne0_w, in_tf), m->ne0_b);
    temb = ggml_add(c, ggml_mul_mat(c, m->ne2_w, ggml_silu(c, temb)), m->ne2_b);

    ten *x = ggml_get_rows(c, m->embed, in_ids);  // [D, T]
    if (in_sc) {
        ten *ev = ggml_cont(c, ggml_transpose(c, ggml_view_2d(c, m->embed, D, VOCAB, m->embed->nb[1], 0)));
        x = ggml_add(c, x, ggml_mul_mat(c, m->self_cond_proj, ggml_mul_mat(c, ev, in_sc)));
    }

    ten *skip = NULL;
    if (m->subpos) {
        x = ggml_reshape_2d(c, ggml_add(c, ggml_reshape_3d(c, x, D, P, Tp), m->subpos), D, T);
        skip = local_residual(&b, x, m->enc_norm, m->enc_conv);
        // compress is an "int8" BitLinear: no fake-quant (see topk()).
        x = ggml_mul_mat(c, m->cmp, norm(&b, ggml_reshape_2d(c, skip, (int64_t) D * P, Tp), m->cmp_norm));
    }

    for (int i = 0; i < m->n_pre; i++) x = block(&b, &m->pre[i], x, temb);
    if (m->is_rdt) x = recurrent(&b, x, temb, n_loops, in_noise, in_sig);
    for (int i = 0; i < m->n_post; i++) x = block(&b, &m->post[i], x, temb);
    x = norm(&b, x, m->final_norm);

    if (m->subpos) {
        ten *e = ggml_reshape_2d(c, ggml_mul_mat(c, m->exp, x), D, T);
        x = ggml_scale(c, ggml_add(c, e, skip), 0.70710678f);  // 2 ** -0.5
        x = local_residual(&b, x, m->dec_norm, m->dec_conv);
        if (m->dec_ffn.up) x = ggml_add(c, x, ffn(&b, &m->dec_ffn, norm(&b, x, m->dec_ffn_norm)));
        x = norm(&b, x, m->out_norm);
    }
    ten *logits = ggml_mul_mat(c, m->embed, x);  // tied head: [rows, T]
    ggml_set_output(logits);

    struct ggml_cgraph *gf = ggml_new_graph_custom(c, GRAPH_SIZE, false);
    ggml_build_forward_expand(gf, logits);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) die("graph allocation failed");

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
        int32_t *pos = malloc(sizeof(int32_t) * Tp);
        for (int i = 0; i < Tp; i++) pos[i] = i;
        set_input(b.pos, pos, sizeof(int32_t) * Tp);
        free(pos);
    }
    float one = 1.0f;
    set_input(b.one, &one, sizeof one);
    if (b.freq_factors) {
        // YaRN ramp from RotaryEmbedding._scaled_inv_freq, as a divisor of theta.
        int n = hd / 2;
        float ff[1024];
        for (int i = 0; i < n && i < 1024; i++) {
            float ramp = n > 1 ? (float) i / (n - 1) : 0.0f;
            ramp = fminf(fmaxf((ramp - 0.25f) / 0.5f, 0.0f), 1.0f);
            ff[i] = 1.0f / ((1.0f - ramp) + ramp / (float) m->rope_factor);
        }
        set_input(b.freq_factors, ff, sizeof(float) * n);
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

    if (ggml_backend_sched_graph_compute(m->sched, gf) != GGML_STATUS_SUCCESS) die("graph compute failed");
    ggml_backend_tensor_get(logits, out, 0, ggml_nbytes(logits));
    ggml_free(c);
}

// ---------------------------------------------------------------------------
// Sampler (yeoubi/sample.py: denoise)
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

static int by_conf(const void *a, const void *b) {
    float x = ((const pi_t *) a)->p, y = ((const pi_t *) b)->p;
    return (x > y) - (x < y);
}

typedef struct {
    const char *model, *device, *prompt, *prompt_file, *dump;
    int length, steps, loops, threads, verbose, remask;  // remask: 0 random, 1 confidence, 2 remdm
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

static int run(model_t *m, const opts_t *o, const unsigned char *prompt, int plen) {
    int P = m->patch, rows = (int) m->embed->ne[1];
    int T = o->length > plen ? o->length : plen + 1;
    T = (T + P - 1) / P * P;  // canvas must fill whole patches; extra bytes are just generated too
    int32_t *ids  = malloc(sizeof(int32_t) * T);
    char *masked  = malloc(T);
    float *logits = malloc(sizeof(float) * rows * T);
    float *sc     = NULL;
    for (int i = 0; i < T; i++) ids[i] = i < plen ? prompt[i] : MASK_ID;

    if (o->dump) {  // parity mode: one forward, raw float32 [T][rows]
        float *z = make_noise(m, T);
        forward(m, ids, T, o->dump_t, select_loops(m, o, o->dump_t, T), NULL, z, logits);
        free(z);
        FILE *f = fopen(o->dump, "wb");
        if (!f) die("cannot write %s", o->dump);
        fwrite(logits, sizeof(float), (size_t) rows * T, f);
        fclose(f);
        fprintf(stderr, "wrote %d x %d logits to %s\n", T, rows, o->dump);
        return 0;
    }

    for (int s = 0; s < o->steps; s++) {
        double tc = acos(1.0 - (1.0 - (double) s / o->steps)) / 1.5707963267948966;
        double tn = acos(1.0 - (1.0 - (double) (s + 1) / o->steps)) / 1.5707963267948966;
        float *z = make_noise(m, T);
        forward(m, ids, T, (float) tc, select_loops(m, o, tc, T), sc, z, logits);
        free(z);
        if (m->self_cond_proj) {
            if (!sc) sc = malloc(sizeof(float) * VOCAB * T);
            for (int i = 0; i < T; i++) softmax(logits + (size_t) i * rows, 1.0f, sc + (size_t) i * VOCAB);
        }

        for (int i = 0; i < T; i++) {
            masked[i] = ids[i] == MASK_ID;
            if (masked[i]) ids[i] = draw(logits + (size_t) i * rows, o->temp, o->top_p);
        }
        if (o->revise > 0 && tc <= o->revise_t_max) {  // _revise_step
            float p[VOCAB];
            for (int i = plen; i < T; i++) {
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
            if (o->remask == 1) {
                pi_t *conf = malloc(sizeof(pi_t) * T);
                float p[VOCAB];
                int n = 0;
                for (int i = plen; i < T; i++) {
                    if (!masked[i]) continue;
                    softmax(logits + (size_t) i * rows, 1.0f, p);
                    conf[n++] = (pi_t) { p[ids[i] < VOCAB ? ids[i] : VOCAB - 1], i };
                }
                qsort(conf, n, sizeof *conf, by_conf);
                int k = (int) nearbyint(n * ratio);
                for (int j = 0; j < k && j < n; j++) ids[conf[j].i] = MASK_ID;
                free(conf);
            } else {
                for (int i = plen; i < T; i++) {
                    double r = rnd();
                    if (masked[i] ? r < ratio : r < sigma) ids[i] = MASK_ID;
                }
            }
        }
        if (o->verbose) {
            int n = 0;
            for (int i = 0; i < T; i++) n += ids[i] == MASK_ID;
            fprintf(stderr, "step %d/%d t=%.3f->%.3f masked=%d\n", s + 1, o->steps, tc, tn, n);
        }
    }

    // Byte IDs only, like decode_byte_ids; the terminal does UTF-8 decoding.
    for (int i = 0; i < T; i++) if (ids[i] < 256) fputc(ids[i], stdout);
    fputc('\n', stdout);
    free(ids); free(masked); free(logits); free(sc);
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
        "  -n, --length N        canvas bytes, prompt included (default 256)\n"
        "  --steps N             denoising steps (default 20)\n"
        "  --temp F              temperature (default 0.9)\n"
        "  --top-p F             nucleus threshold (default 0.95)\n"
        "  --seed N              RNG seed (default 42)\n"
        "  --loops N             recurrent loop override (RDT)\n"
        "  --remask MODE         random | confidence | remdm (default random)\n"
        "  --eta F               remdm cap (default 0.04)\n"
        "  --revise F            overwrite visible bytes above this confidence (default off)\n"
        "  --revise-t-max F      only revise when t <= F (default 0.5)\n"
        "  --act a4|a8           a4 = trained INT4/TopK activations (default), a8 = none\n"
        "  --dump-logits F       one forward at --t, write float32 logits, exit\n"
        "  --t F                 noise level for --dump-logits (default 1.0)\n"
        "  -v, --verbose         per-step progress on stderr\n");
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
    int act_a4 = 1, list = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
#define ARG(s, l) ((!strcmp(a, s) || !strcmp(a, l)) && more)
        if      (ARG("-m", "--model"))       o.model = argv[++i];
        else if (ARG("-d", "--device"))      o.device = argv[++i];
        else if (ARG("-t", "--threads"))     o.threads = atoi(argv[++i]);
        else if (ARG("-p", "--prompt"))      o.prompt = argv[++i];
        else if (ARG("-f", "--prompt-file")) o.prompt_file = argv[++i];
        else if (ARG("-n", "--length"))      o.length = atoi(argv[++i]);
        else if (ARG("--steps", "--steps"))  o.steps = atoi(argv[++i]);
        else if (ARG("--temp", "--temp"))    o.temp = (float) atof(argv[++i]);
        else if (ARG("--top-p", "--top-p"))  o.top_p = (float) atof(argv[++i]);
        else if (ARG("--seed", "--seed"))    o.seed = strtoull(argv[++i], NULL, 10);
        else if (ARG("--loops", "--loops"))  o.loops = atoi(argv[++i]);
        else if (ARG("--eta", "--eta"))      o.eta = (float) atof(argv[++i]);
        else if (ARG("--revise", "--revise")) o.revise = (float) atof(argv[++i]);
        else if (ARG("--revise-t-max", "--revise-t-max")) o.revise_t_max = (float) atof(argv[++i]);
        else if (ARG("--dump-logits", "--dump-logits")) o.dump = argv[++i];
        else if (ARG("--t", "--t"))          o.dump_t = (float) atof(argv[++i]);
        else if (ARG("--act", "--act"))      act_a4 = strcmp(argv[++i], "a8") != 0;
        else if (ARG("--remask", "--remask")) {
            const char *r = argv[++i];
            o.remask = !strcmp(r, "confidence") ? 1 : !strcmp(r, "remdm") ? 2 : !strcmp(r, "random") ? 0 : -1;
            if (o.remask < 0) usage();
        }
        else if (!strcmp(a, "--list-devices")) list = 1;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) o.verbose = 1;
        else usage();
#undef ARG
    }
    if (o.steps < 1 || o.length < 1 || o.loops > MAX_LOOPS) usage();

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
    m.act_a4 = act_a4;
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

    model_load(&m, o.model);
    if (m.is_rdt && m.max_loops > MAX_LOOPS) die("max_loop_iters > %d", MAX_LOOPS);
    // Unsupported ops on the main device fall back to the CPU (always last).
    ggml_backend_t backends[2] = { m.backend, m.cpu };
    m.sched = ggml_backend_sched_new(backends, NULL, on_cpu ? 1 : 2, GRAPH_SIZE, false, false);

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
    int rc = run(&m, &o, prompt, plen);

    ggml_backend_sched_free(m.sched);
    ggml_backend_buffer_free(m.wbuf);
    ggml_free(m.wctx);
    if (!on_cpu) ggml_backend_free(m.cpu);
    ggml_backend_free(m.backend);
    return rc;
}
