#include "audio_adaptor.hpp"

#include "backend.hpp"
#include "common.hpp"
#include "ggml_extend.hpp"

#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace mt {

namespace {
ggml_tensor* req(ModelLoader& m, const char* name) {
    ggml_tensor* t = m.tensor(name);
    if (!t) MT_LOGE("audio_adaptor: missing tensor '%s'", name);
    return t;
}
}  // namespace

AudioAdaptor::AudioAdaptor(ModelLoader& m) {
    const Config& c = m.config();
    merge_  = c.audio_merge_size;
    in_dim_ = c.adaptor_input_dim;
    hidden_ = c.text_hidden;
    eps_    = c.adaptor_norm_eps;

    fc1_w_ = req(m, "adaptor.fc1.w");
    fc1_b_ = req(m, "adaptor.fc1.b");
    fc2_w_ = req(m, "adaptor.fc2.w");
    fc2_b_ = req(m, "adaptor.fc2.b");
    ln_w_  = req(m, "adaptor.ln.w");
    ln_b_  = req(m, "adaptor.ln.b");
}

void AudioAdaptor::apply(const std::vector<float>& enc, int T, int D,
                         std::vector<float>& out, int& N, int& H) const {
    const int Ttrim = (T / merge_) * merge_;
    N = Ttrim / merge_;
    H = hidden_;
    out.clear();
    if (N <= 0) return;

    const size_t max_nodes = 256;
    const size_t mem = ggml_tensor_overhead() * max_nodes +
                       ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> buf(mem);
    GgmlCtxPtr ctx_ptr = make_ctx_buf(buf.data(), mem, /*no_alloc=*/true);
    ggml_context* ctx = ctx_ptr.get();

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, max_nodes, false);

    // Encoder output as ne=[D, T] (feature fastest).
    ggml_tensor* enc_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
    ggml_set_name(enc_in, "enc");
    ggml_set_input(enc_in);

    // Trim to [D, Ttrim] (contiguous view of leading frames) then reshape to
    // [D*merge, N] — groups `merge` consecutive frames' features into one vector
    // in the order [frame0, frame1, ...] (torch reshape(B, N, D*merge) parity).
    ggml_tensor* trimmed = enc_in;
    if (Ttrim != T) {
        trimmed = ggml_view_2d(ctx, enc_in, D, Ttrim, enc_in->nb[1], 0);
    }
    trimmed = ggml_cont(ctx, trimmed);
    ggml_tensor* cur = ggml_reshape_2d(ctx, trimmed, (int64_t)D * merge_, N);  // [D*merge, N]

    cur = linear(ctx, fc1_w_, fc1_b_, cur);            // [hidden, N]
    cur = ggml_silu(ctx, cur);
    cur = linear(ctx, fc2_w_, fc2_b_, cur);            // [hidden, N]
    cur = layer_norm(ctx, cur, ln_w_, ln_b_, eps_);    // [hidden, N]

    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    const bool ok = compute_graph_with_inputs(gf, [&]() {
        ggml_backend_tensor_set(enc_in, enc.data(),
                                0, (size_t)D * (size_t)T * sizeof(float));
    });
    if (!ok) {
        MT_LOGE("audio_adaptor: graph compute failed");
        out.clear();
        N = H = 0;
        return;
    }

    H = (int)cur->ne[0];
    N = (int)cur->ne[1];
    read_tensor_f32(cur, &out);
}

}  // namespace mt
