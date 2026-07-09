#include "qwen3_decoder.hpp"

#include "backend.hpp"
#include "ggml_extend.hpp"
#include "common.hpp"   // MT_LOGE

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace mt {

Qwen3Decoder::~Qwen3Decoder() {
    if (kv_buffer_) ggml_backend_buffer_free(kv_buffer_);
}

bool Qwen3Decoder::load(const ModelLoader& m, int max_seq) {
    m_        = &m;
    max_seq_  = max_seq;
    past_len_ = 0;

    // Hparams come from mt::Config (mtd.text.*), NOT from qwen3.* KV: the
    // converter did not write the qwen3.* metadata block.
    const Config& c = m.config();
    hp_.hidden       = c.text_hidden;
    hp_.n_heads      = c.text_heads;
    hp_.n_kv_heads   = c.text_kv_heads;
    hp_.head_dim     = c.text_head_dim;   // 128 — independent of hidden/n_heads
    hp_.intermediate = c.text_ffn;
    hp_.n_layers     = c.text_layers;
    hp_.text_vocab   = c.text_vocab;
    hp_.rope_base    = c.text_rope_theta;  // 1e6
    hp_.rms_eps      = c.text_rms_eps;     // 1e-6
    hp_.use_rope     = true;

    if (hp_.hidden <= 0 || hp_.n_layers <= 0 || hp_.n_heads <= 0 ||
        hp_.n_kv_heads <= 0 || hp_.head_dim <= 0 || max_seq_ <= 0) {
        MT_LOGE("Qwen3Decoder: bad hparams");
        return false;
    }

    layers_.assign(hp_.n_layers, Qwen3Layer{});
    for (int i = 0; i < hp_.n_layers; ++i) {
        if (!qwen3_load_layer(m, "qwen3", i, &layers_[i])) {
            MT_LOGE("Qwen3Decoder: failed to load layer %d", i);
            return false;
        }
    }

    output_norm_ = m.tensor("qwen3.output_norm.weight");
    if (!output_norm_) { MT_LOGE("Qwen3Decoder: missing qwen3.output_norm.weight"); return false; }

    token_embd_ = m.tensor("token_embd.weight");   // [hidden, vocab] — tied lm_head
    if (!token_embd_) { MT_LOGE("Qwen3Decoder: missing token_embd.weight"); return false; }

    // Persistent device-resident K/V cache: 2 tensors per layer,
    // [head_dim, n_kv_heads, max_seq, 1].
    const int L = hp_.n_layers;
    kv_ctx_ = make_ctx(ggml_tensor_overhead() * (size_t)(2 * L) + 1024, /*no_alloc=*/true);
    if (!kv_ctx_) return false;
    k_cache_.assign(L, nullptr);
    v_cache_.assign(L, nullptr);
    for (int l = 0; l < L; ++l) {
        k_cache_[l] = ggml_new_tensor_4d(kv_ctx_.get(), GGML_TYPE_F32,
                                         hp_.head_dim, hp_.n_kv_heads, max_seq_, 1);
        v_cache_[l] = ggml_new_tensor_4d(kv_ctx_.get(), GGML_TYPE_F32,
                                         hp_.head_dim, hp_.n_kv_heads, max_seq_, 1);
    }
    kv_buffer_ = allocate_ctx_tensors(kv_ctx_.get());
    if (!kv_buffer_) { MT_LOGE("Qwen3Decoder: KV cache alloc failed"); return false; }

    // Reused metadata-ctx buffer for every run() graph (avoids a big malloc
    // per step). Sized to the node budget.
    scratch_.resize(ggml_tensor_overhead() * 2 * 4096
                  + ggml_graph_overhead_custom(4096, false)
                  + (1u << 20));
    return true;
}

void Qwen3Decoder::reset() {
    past_len_ = 0;  // views are bounded by past_len_; stale cache bytes are never read
}

bool Qwen3Decoder::run(const std::vector<float>& embeds, int T,
                       std::vector<float>* out_hidden) {
    const int L    = hp_.n_layers;
    const int H    = hp_.hidden;
    const int past = past_len_;
    const int kv   = past + T;

    if (embeds.size() < (size_t)H * T) { MT_LOGE("Qwen3Decoder: embeds too small"); return false; }
    if (past + T > max_seq_) { MT_LOGE("Qwen3Decoder: sequence exceeds max_seq"); return false; }

    auto cctx = make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_set_input(x);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);
    // Causal mask only needed when T>1 (prefill). For a T=1 decode step every
    // cached key is causally valid, so mask=null (no bias).
    struct ggml_tensor* mask = nullptr;
    if (T > 1) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv, T);  // ne0=key, ne1=query
        ggml_set_input(mask);
    }

    auto* gf = ggml_new_graph_custom(ctx, 4096, false);

    struct ggml_tensor* h = x;
    for (int l = 0; l < L; ++l) {
        auto lo = qwen3_layer_forward(ctx, h, pos, mask, /*k_past=*/nullptr, /*v_past=*/nullptr,
                                      layers_[l], hp_, gf, k_cache_[l], v_cache_[l], past);
        h = lo.y;
    }

    // Final RMSNorm on ALL positions -> [H, T].
    struct ggml_tensor* y = ggml_mul(ctx, ggml_rms_norm(ctx, h, hp_.rms_eps), output_norm_);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    const float ninf = -std::numeric_limits<float>::infinity();
    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, embeds.data(), 0, (size_t)H * T * sizeof(float));

        std::vector<int32_t> p(T);
        for (int i = 0; i < T; ++i) p[i] = past + i;
        ggml_backend_tensor_set(pos, p.data(), 0, (size_t)T * sizeof(int32_t));

        if (mask) {
            // mask[query i][key j]: attend iff key j <= query abs pos past+i.
            std::vector<float> mvec((size_t)kv * T);
            for (int i = 0; i < T; ++i)
                for (int j = 0; j < kv; ++j)
                    mvec[(size_t)i * kv + j] = (j <= past + i) ? 0.0f : ninf;
            ggml_backend_tensor_set(mask, mvec.data(), 0, mvec.size() * sizeof(float));
        }
    };

    if (!compute_graph_with_inputs(gf, set_inputs)) return false;

    past_len_ = kv;
    out_hidden->resize((size_t)H * T);
    ggml_backend_tensor_get(y, out_hidden->data(), 0, (size_t)H * T * sizeof(float));
    return true;
}

bool Qwen3Decoder::prefill(const std::vector<float>& embeds, int S,
                           std::vector<float>* all_hidden) {
    reset();
    return run(embeds, S, all_hidden);
}

std::vector<float> Qwen3Decoder::decode_one(const std::vector<float>& embed) {
    std::vector<float> hidden;
    if (!run(embed, 1, &hidden)) hidden.clear();
    return hidden;
}

std::vector<float> Qwen3Decoder::logits_from_hidden(const std::vector<float>& hidden_row) {
    std::vector<float> out;
    const int H = hp_.hidden;
    if ((int)hidden_row.size() < H || !token_embd_) return out;

    // token_embd_ ne=[hidden, vocab]; mul_mat with [hidden,1] -> [vocab,1].
    size_t buf_sz = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + (1u << 16);
    std::vector<uint8_t> buf(buf_sz);
    auto cctx = make_ctx_buf(buf.data(), buf.size(), /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();

    struct ggml_tensor* hin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
    ggml_set_input(hin);
    struct ggml_tensor* logits = ggml_mul_mat(ctx, token_embd_, hin);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ggml_set_output(logits);

    auto* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);

    auto set_inputs = [&]() {
        ggml_backend_tensor_set(hin, hidden_row.data(), 0, (size_t)H * sizeof(float));
    };
    if (!compute_graph_with_inputs(gf, set_inputs)) return out;

    read_tensor_f32(logits, &out);
    return out;
}

}  // namespace mt
