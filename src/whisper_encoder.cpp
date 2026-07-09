#include "whisper_encoder.hpp"

#include "backend.hpp"
#include "common.hpp"
#include "ggml_extend.hpp"

#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace mt {

namespace {
// Fetch a required encoder tensor by name; logs if missing.
ggml_tensor* req(ModelLoader& m, const std::string& name) {
    ggml_tensor* t = m.tensor(name);
    if (!t) MT_LOGE("whisper_encoder: missing tensor '%s'", name.c_str());
    return t;
}

// 1-D convolution with "same" (half-kernel) padding, computed entirely in F32.
// ggml_conv_1d_ph hardcodes an F16 im2col (requiring an F16 kernel); our conv
// weights are F32, so we replicate ggml_conv_1d with an F32 im2col to preserve
// precision for the parity gate.
// a: kernel ne=[K, IC, OC]; b: input ne=[L, IC] -> result ne=[OL, OC].
ggml_tensor* conv1d_ph_f32(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b,
                           int s0, int d0) {
    const int p0 = (int)a->ne[0] / 2;
    ggml_tensor* im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0,
                                      /*is_2D=*/false, GGML_TYPE_F32);  // [N, OL, IC*K]
    ggml_tensor* result = ggml_mul_mat(
        ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2]));
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]);  // [OL, OC, N]
    return result;
}
}  // namespace

WhisperEncoder::WhisperEncoder(ModelLoader& m) {
    const Config& c = m.config();
    d_model_     = c.audio_d_model;
    n_layers_    = c.audio_layers;
    n_heads_     = c.audio_heads;
    ffn_         = c.audio_ffn;
    max_src_pos_ = c.audio_max_src_pos;

    conv1_w_   = req(m, "enc.conv1.w");
    conv1_b_   = req(m, "enc.conv1.b");
    conv2_w_   = req(m, "enc.conv2.w");
    conv2_b_   = req(m, "enc.conv2.b");
    pos_embd_  = req(m, "enc.pos_embd");
    ln_post_w_ = req(m, "enc.ln_post.w");
    ln_post_b_ = req(m, "enc.ln_post.b");

    layers_.resize(n_layers_);
    for (int i = 0; i < n_layers_; ++i) {
        const std::string p = "enc.blk." + std::to_string(i) + ".";
        WhisperLayer& L = layers_[i];
        L.attn_ln_w = req(m, p + "attn_ln.w");
        L.attn_ln_b = req(m, p + "attn_ln.b");
        L.q_w       = req(m, p + "attn_q.w");
        L.q_b       = req(m, p + "attn_q.b");
        L.k_w       = req(m, p + "attn_k.w");   // no bias
        L.v_w       = req(m, p + "attn_v.w");
        L.v_b       = req(m, p + "attn_v.b");
        L.o_w       = req(m, p + "attn_out.w");
        L.o_b       = req(m, p + "attn_out.b");
        L.ffn_ln_w  = req(m, p + "ffn_ln.w");
        L.ffn_ln_b  = req(m, p + "ffn_ln.b");
        L.fc1_w     = req(m, p + "ffn_1.w");
        L.fc1_b     = req(m, p + "ffn_1.b");
        L.fc2_w     = req(m, p + "ffn_2.w");
        L.fc2_b     = req(m, p + "ffn_2.b");
    }
}

void WhisperEncoder::encode(const std::vector<float>& mel, int n_mels, int n_frames,
                            std::vector<float>& out, int& out_T, int& out_D) const {
    (void)n_mels;
    const int d  = d_model_;
    const int H  = n_heads_;
    const int hd = d / H;

    const size_t max_nodes = 8192;
    const size_t mem = ggml_tensor_overhead() * max_nodes +
                       ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> buf(mem);
    GgmlCtxPtr ctx_ptr = make_ctx_buf(buf.data(), mem, /*no_alloc=*/true);
    ggml_context* ctx = ctx_ptr.get();

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, max_nodes, false);

    // mel input: ne=[L=n_frames, IC=n_mels] (t fastest == L fastest).
    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames, n_mels);
    ggml_set_name(mel_in, "mel");
    ggml_set_input(mel_in);

    // conv stem: conv1 (stride 1) -> +bias -> gelu_erf; conv2 (stride 2) -> +bias -> gelu_erf.
    ggml_tensor* cur = conv1d_ph_f32(ctx, conv1_w_, mel_in, 1, 1);     // [n_frames, d]
    cur = ggml_add(ctx, cur, ggml_reshape_2d(ctx, conv1_b_, 1, d));
    cur = ggml_gelu_erf(ctx, cur);
    cur = conv1d_ph_f32(ctx, conv2_w_, cur, 2, 1);                     // [1500, d]
    cur = ggml_add(ctx, cur, ggml_reshape_2d(ctx, conv2_b_, 1, d));
    cur = ggml_gelu_erf(ctx, cur);

    const int T = (int)cur->ne[0];  // 1500

    // transpose to [d_model, T] (feature fastest), add positional embeddings.
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));   // [d, T]
    cur = ggml_add(ctx, cur, pos_embd_);              // enc.pos_embd ne=[d, T]

    const float kq_scale = 1.0f / std::sqrt((float)hd);

    for (int il = 0; il < n_layers_; ++il) {
        const WhisperLayer& L = layers_[il];

        ggml_tensor* res = cur;
        ggml_tensor* x   = layer_norm(ctx, cur, L.attn_ln_w, L.attn_ln_b, 1e-5f);

        ggml_tensor* q = linear(ctx, L.q_w, L.q_b, x);       // [d, T]
        ggml_tensor* k = linear(ctx, L.k_w, nullptr, x);     // [d, T] (no bias)
        ggml_tensor* v = linear(ctx, L.v_w, L.v_b, x);       // [d, T]

        // [d,T] -> [hd,H,T] -> [hd,T,H]
        ggml_tensor* Q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, hd, H, T), 0, 2, 1, 3);
        ggml_tensor* K = ggml_permute(ctx, ggml_reshape_3d(ctx, k, hd, H, T), 0, 2, 1, 3);

        ggml_tensor* scores = ggml_mul_mat(ctx, K, Q);       // [T, T, H]
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        // no causal mask (bidirectional encoder); scale = 1/sqrt(head_dim).
        ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, nullptr, kq_scale, 0.0f);

        // V: [hd,H,T] -> [T,hd,H]
        ggml_tensor* V = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, hd, H, T), 1, 2, 0, 3));
        ggml_tensor* KQV = ggml_mul_mat(ctx, V, attn);       // [hd, T, H]
        ggml_tensor* merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);  // [hd, H, T]
        x = ggml_cont_2d(ctx, merged, d, T);                 // [d, T]

        x   = linear(ctx, L.o_w, L.o_b, x);
        cur = ggml_add(ctx, res, x);

        ggml_tensor* res2 = cur;
        x = layer_norm(ctx, cur, L.ffn_ln_w, L.ffn_ln_b, 1e-5f);
        x = linear(ctx, L.fc1_w, L.fc1_b, x);
        x = ggml_gelu_erf(ctx, x);
        x = linear(ctx, L.fc2_w, L.fc2_b, x);
        cur = ggml_add(ctx, res2, x);
    }

    // final LayerNorm.
    cur = layer_norm(ctx, cur, ln_post_w_, ln_post_b_, 1e-5f);
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    const bool ok = compute_graph_with_inputs(gf, [&]() {
        ggml_backend_tensor_set(mel_in, mel.data(), 0, mel.size() * sizeof(float));
    });
    if (!ok) {
        MT_LOGE("whisper_encoder: graph compute failed");
        out.clear();
        out_T = out_D = 0;
        return;
    }

    out_D = (int)cur->ne[0];   // d_model
    out_T = (int)cur->ne[1];   // T
    read_tensor_f32(cur, &out);
}

}  // namespace mt
