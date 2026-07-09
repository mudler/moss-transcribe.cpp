#include "generate.hpp"

#include "backend.hpp"
#include "common.hpp"
#include "ggml_extend.hpp"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

namespace mt {

// Dequantize `n_ids` rows of token_embd via ggml_get_rows. Works for ANY
// token_embd type (F32, F16, Q8_0, K-quants): get_rows always returns F32
// rows. For an F32 token_embd this is bit-identical to a raw column copy.
// token_embd ne=[hidden, vocab]; result is [hidden, n_ids] laid out
// feature-fastest, so out[p*hidden + h] is feature h of ids[p]. Returns
// false on failure (out is left empty by the caller on failure).
static bool get_rows_f32(struct ggml_tensor* tok, const int32_t* ids,
                         int n_ids, int hidden, std::vector<float>* out) {
    if (!tok || !ids || n_ids <= 0 || hidden <= 0 || !out) return false;

    const size_t max_nodes = 8;
    size_t buf_sz = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead() +
                    (1u << 16);
    std::vector<uint8_t> buf(buf_sz);
    GgmlCtxPtr cctx = make_ctx_buf(buf.data(), buf.size(), /*no_alloc=*/true);
    ggml_context* ctx = cctx.get();
    if (!ctx) return false;

    struct ggml_tensor* idt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ids);
    ggml_set_input(idt);
    struct ggml_tensor* rows = ggml_get_rows(ctx, tok, idt);  // [hidden, n_ids] F32
    ggml_set_output(rows);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rows);

    const bool ok = compute_graph_with_inputs(gf, [&]() {
        ggml_backend_tensor_set(idt, ids, 0, (size_t)n_ids * sizeof(int32_t));
    });
    if (!ok) return false;
    return read_tensor_f32(rows, out);
}

std::vector<float> fuse_embeds(ModelLoader& m,
                               const std::vector<int32_t>& input_ids,
                               const std::vector<float>& audio_embeds,
                               int n_audio, int hidden, int audio_token_id) {
    std::vector<float> out;
    if (hidden <= 0) {
        MT_LOGE("fuse_embeds: invalid hidden=%d", hidden);
        return out;
    }
    struct ggml_tensor* tok = m.tensor("token_embd.weight");
    if (!tok) {
        MT_LOGE("fuse_embeds: missing token_embd.weight");
        return out;
    }
    // token_embd.weight ne=[hidden, vocab]: column t is the embedding of token
    // id t, laid out feature-fastest (token_embd_data[t*hidden + h]).
    if ((int)tok->ne[0] != hidden) {
        MT_LOGE("fuse_embeds: token_embd hidden %lld != %d",
                (long long)tok->ne[0], hidden);
        return out;
    }
    const int64_t vocab = tok->ne[1];
    const size_t  seq   = input_ids.size();
    if (seq == 0) return out;

    // Validate ids before the get_rows lookup (out-of-range ids would read
    // garbage rows).
    for (size_t p = 0; p < seq; ++p) {
        int32_t t = input_ids[p];
        if (t < 0 || t >= vocab) {
            MT_LOGE("fuse_embeds: input_ids[%zu]=%d out of range [0,%lld)",
                    p, t, (long long)vocab);
            return out;
        }
    }

    // 1) Embed lookup via ggml_get_rows: dequantizes rows for ANY token_embd
    //    type (F32/F16/quant) and returns F32 rows [hidden, seq]. On an F32
    //    token_embd this is bit-identical to the old raw column copy.
    if (!get_rows_f32(tok, input_ids.data(), (int)seq, hidden, &out)) {
        MT_LOGE("fuse_embeds: get_rows lookup failed");
        out.clear();
        return out;
    }

    // 2) Audio injection (masked_scatter): walk positions in increasing index;
    //    the k-th position with id==audio_token_id gets audio_embeds row k.
    int k = 0;
    for (size_t p = 0; p < seq; ++p) {
        if (input_ids[p] != audio_token_id) continue;
        if (k >= n_audio) {
            MT_LOGE("fuse_embeds: more audio positions than audio rows "
                    "(n_audio=%d)", n_audio);
            out.clear();
            return out;
        }
        const float* src = audio_embeds.data() + (size_t)k * (size_t)hidden;
        float* dst = out.data() + p * (size_t)hidden;
        for (int h = 0; h < hidden; ++h) dst[h] = src[h];
        ++k;
    }
    if (k != n_audio) {
        MT_LOGW("fuse_embeds: consumed %d of %d audio rows", k, n_audio);
    }
    return out;
}

std::vector<float> embed_token(ModelLoader& m, int32_t t, int hidden) {
    std::vector<float> out;
    if (hidden <= 0) return out;
    struct ggml_tensor* tok = m.tensor("token_embd.weight");
    if (!tok) { MT_LOGE("embed_token: missing token_embd.weight"); return out; }
    if ((int)tok->ne[0] != hidden) {
        MT_LOGE("embed_token: token_embd hidden %lld != %d", (long long)tok->ne[0], hidden);
        return out;
    }
    const int64_t vocab = tok->ne[1];
    if (t < 0 || t >= vocab) {
        MT_LOGE("embed_token: id %d out of range [0,%lld)", (int)t, (long long)vocab);
        return out;
    }
    // Single-row lookup via ggml_get_rows: dequantizes for ANY token_embd type
    // (F32/F16/quant), F32 result [hidden]. Bit-identical to the old raw read
    // on an F32 token_embd.
    if (!get_rows_f32(tok, &t, 1, hidden, &out)) {
        MT_LOGE("embed_token: get_rows lookup failed for id %d", (int)t);
        out.clear();
    }
    return out;
}

// argmax with FIRST-index-on-tie semantics (torch argmax): strict >.
static int argmax_first(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int)v.size(); ++i) if (v[i] > v[best]) best = i;
    return best;
}

std::vector<int32_t> greedy_generate(Qwen3Decoder& dec, ModelLoader& m,
                                     const std::vector<float>& fused, int seq,
                                     int max_new, int eos) {
    std::vector<int32_t> ids;
    const int H = dec.hidden();
    if (H <= 0 || seq <= 0 || max_new <= 0) {
        MT_LOGE("greedy_generate: bad args (H=%d seq=%d max_new=%d)", H, seq, max_new);
        return ids;
    }

    std::vector<float> hid;
    if (!dec.prefill(fused, seq, &hid)) { MT_LOGE("greedy_generate: prefill failed"); return ids; }
    if ((int)hid.size() < H * seq) { MT_LOGE("greedy_generate: short prefill hidden"); return ids; }

    // Logits from the last prefilled position.
    std::vector<float> last(hid.end() - H, hid.end());
    std::vector<float> logits = dec.logits_from_hidden(last);
    if (logits.empty()) { MT_LOGE("greedy_generate: logits failed"); return ids; }

    ids.reserve((size_t)max_new);
    for (;;) {
        int t = argmax_first(logits);
        ids.push_back(t);
        if (t == eos) break;
        if ((int)ids.size() >= max_new) break;

        std::vector<float> emb = embed_token(m, t, H);
        if (emb.empty()) { MT_LOGE("greedy_generate: embed failed @%d", t); break; }
        std::vector<float> h1 = dec.decode_one(emb);
        if ((int)h1.size() < H) { MT_LOGE("greedy_generate: decode_one failed"); break; }
        logits = dec.logits_from_hidden(h1);
        if (logits.empty()) { MT_LOGE("greedy_generate: logits failed"); break; }
    }
    return ids;
}

}  // namespace mt
