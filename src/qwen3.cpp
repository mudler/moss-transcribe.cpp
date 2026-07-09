#include "qwen3.hpp"
#include "backend.hpp"

#include <cmath>

namespace mt {

namespace {

constexpr int kQwen3RopeMode = GGML_ROPE_TYPE_NEOX;

inline struct ggml_tensor* maybe_cont(struct ggml_context* ctx, struct ggml_tensor* t) {
    // ggml_mul_mat asserts !ggml_is_transposed(a) (nb[0] <= nb[1]). A transpose
    // of a [hd, seq_kv=1, ...] tensor has ne0=1, which ggml_is_contiguous still
    // reports as contiguous even though nb[0] > nb[1] — so check transposed too.
    // This is the single-key depth-decode case (seq_kv == 1, no KV cache).
    return (ggml_is_contiguous(t) && !ggml_is_transposed(t)) ? t : ggml_cont(ctx, t);
}

// RMSNorm over ne0, scaled by weight (broadcasts over the higher dims).
inline struct ggml_tensor* rms_norm(struct ggml_context* ctx, struct ggml_tensor* x,
                                    struct ggml_tensor* w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

}  // namespace

bool qwen3_load_layer(const ModelLoader& m, const std::string& prefix, int i, Qwen3Layer* out) {
    if (!out) return false;
    const std::string b = prefix + ".blk." + std::to_string(i) + ".";
    auto get = [&](const std::string& name, struct ggml_tensor** dst) -> bool {
        struct ggml_tensor* t = m.tensor(b + name);
        if (!t) return false;
        *dst = t;
        return true;
    };
    bool ok = true;
    ok &= get("attn_norm.weight",   &out->attn_norm);
    ok &= get("attn_q.weight",      &out->attn_q);
    ok &= get("attn_k.weight",      &out->attn_k);
    ok &= get("attn_v.weight",      &out->attn_v);
    ok &= get("attn_o.weight",      &out->attn_o);
    ok &= get("attn_q_norm.weight", &out->q_norm);
    ok &= get("attn_k_norm.weight", &out->k_norm);
    ok &= get("ffn_norm.weight",    &out->ffn_norm);
    ok &= get("ffn_gate.weight",    &out->ffn_gate);
    ok &= get("ffn_up.weight",      &out->ffn_up);
    ok &= get("ffn_down.weight",    &out->ffn_down);
    return ok;
}

bool qwen3_load_layer(const ModelLoader& m, int i, Qwen3Layer* out) {
    return qwen3_load_layer(m, "qwen3", i, out);
}

Qwen3LayerOut qwen3_layer_forward(struct ggml_context* ctx, struct ggml_tensor* x,
                                  struct ggml_tensor* pos, struct ggml_tensor* mask,
                                  struct ggml_tensor* k_past, struct ggml_tensor* v_past,
                                  const Qwen3Layer& w, const Qwen3Hparams& hp,
                                  struct ggml_cgraph* gf,
                                  struct ggml_tensor* k_cache, struct ggml_tensor* v_cache,
                                  int past_seq) {
    const int hd     = hp.head_dim;
    const int n_h    = hp.n_heads;
    const int n_kv_h = hp.n_kv_heads;
    const float eps  = hp.rms_eps;

    const int64_t n_tokens = x->ne[1];
    const int64_t n_batch  = x->ne[2] > 0 ? x->ne[2] : 1;

    // ---- attention pre-norm ----
    struct ggml_tensor* xn = rms_norm(ctx, x, w.attn_norm, eps);

    // ---- q, k, v (no bias in Qwen3) ----
    struct ggml_tensor* q = ggml_mul_mat(ctx, w.attn_q, xn);
    struct ggml_tensor* k = ggml_mul_mat(ctx, w.attn_k, xn);
    struct ggml_tensor* v = ggml_mul_mat(ctx, w.attn_v, xn);

    // Reshape to [hd, n_h, seq, batch] and [hd, n_kv_h, seq, batch].
    q = ggml_reshape_4d(ctx, q, hd, n_h,    n_tokens, n_batch);
    k = ggml_reshape_4d(ctx, k, hd, n_kv_h, n_tokens, n_batch);
    v = ggml_reshape_4d(ctx, v, hd, n_kv_h, n_tokens, n_batch);

    // ---- per-head RMSNorm on Q and K over head_dim (ne0), BEFORE RoPE ----
    q = rms_norm(ctx, q, w.q_norm, eps);
    k = rms_norm(ctx, k, w.k_norm, eps);

    // ---- RoPE NEOX on Q and K (skipped when use_rope=false) ----
    if (hp.use_rope) {
        q = ggml_rope_ext(ctx, q, pos, /*freq_factors=*/nullptr,
                          hd, kQwen3RopeMode, /*n_ctx_orig=*/0,
                          hp.rope_base, /*freq_scale=*/1.0f,
                          /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                          /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
        k = ggml_rope_ext(ctx, k, pos, /*freq_factors=*/nullptr,
                          hd, kQwen3RopeMode, 0,
                          hp.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    Qwen3LayerOut out;
    struct ggml_tensor* k_used;
    struct ggml_tensor* v_used;
    if (k_cache) {
        // ---- device-resident cache: store new columns, read prefix by view ----
        const int64_t kv = (int64_t)past_seq + n_tokens;
        struct ggml_tensor* k_dst = ggml_view_4d(ctx, k_cache, hd, n_kv_h, n_tokens, 1,
            k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], (size_t)past_seq * k_cache->nb[2]);
        struct ggml_tensor* v_dst = ggml_view_4d(ctx, v_cache, hd, n_kv_h, n_tokens, 1,
            v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], (size_t)past_seq * v_cache->nb[2]);
        out.k_store = ggml_cpy(ctx, k, k_dst);
        out.v_store = ggml_cpy(ctx, v, v_dst);
        // Expand the stores NOW so they execute before this layer's attention
        // read of the same buffer (ordering by node insertion order).
        ggml_build_forward_expand(gf, out.k_store);
        ggml_build_forward_expand(gf, out.v_store);
        k_used = ggml_view_4d(ctx, k_cache, hd, n_kv_h, kv, 1,
            k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], 0);
        v_used = ggml_view_4d(ctx, v_cache, hd, n_kv_h, kv, 1,
            v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], 0);
    } else {
        // ---- additive path: concat past K/V along the sequence dim (axis 2) ----
        struct ggml_tensor* k_full = k_past ? ggml_concat(ctx, k_past, k, /*dim=*/2) : k;
        struct ggml_tensor* v_full = v_past ? ggml_concat(ctx, v_past, v, /*dim=*/2) : v;
        k_used = k_full;
        v_used = v_full;
        out.k_full = ggml_cont(ctx, k_full);
        out.v_full = ggml_cont(ctx, v_full);
    }

    // ---- eager GQA attention (shared by both paths) ----
    struct ggml_tensor* q_p = ggml_permute(ctx, q,      0, 2, 1, 3);  // [hd, seq, n_h, b]
    struct ggml_tensor* k_p = ggml_permute(ctx, k_used, 0, 2, 1, 3);  // [hd, seq_kv, n_kv, b]
    struct ggml_tensor* v_p = ggml_permute(ctx, v_used, 0, 2, 1, 3);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    struct ggml_tensor* scores = ggml_mul_mat(ctx, k_p, q_p);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    struct ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, /*max_bias=*/0.0f);

    struct ggml_tensor* v_t = maybe_cont(ctx, ggml_transpose(ctx, v_p));  // [seq_kv, hd, n_kv, b]
    struct ggml_tensor* o   = ggml_mul_mat(ctx, v_t, attn);

    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont_2d(ctx, o, n_h * hd, n_tokens * n_batch);
    if (n_batch > 1) o = ggml_reshape_3d(ctx, o, n_h * hd, n_tokens, n_batch);

    o = ggml_mul_mat(ctx, w.attn_o, o);
    struct ggml_tensor* h = ggml_add(ctx, x, o);

    // ---- FFN: SwiGLU = down( silu(gate(x)) * up(x) ) ----
    struct ggml_tensor* hn = rms_norm(ctx, h, w.ffn_norm, eps);
    struct ggml_tensor* g  = ggml_mul_mat(ctx, w.ffn_gate, hn);
    struct ggml_tensor* u  = ggml_mul_mat(ctx, w.ffn_up,   hn);
    struct ggml_tensor* f  = ggml_mul_mat(ctx, w.ffn_down, ggml_mul(ctx, ggml_silu(ctx, g), u));

    out.y = ggml_add(ctx, h, f);
    return out;
}

}  // namespace mt
