#include "generate.hpp"

#include "common.hpp"

#include "ggml-backend.h"

#include <cstddef>

namespace mt {

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
    // Raw F32 reads below assume the tensor is GGML_TYPE_F32. token_embd is
    // huge (~155M elems) and is never promoted to F32, so a non-F32 (f16/quant)
    // GGUF would make ggml_backend_tensor_get read garbage. Fail loud instead.
    if (tok->type != GGML_TYPE_F32) {
        MT_LOGE("fuse_embeds: token_embd type %d != GGML_TYPE_F32 (F32 GGUF required)",
                (int)tok->type);
        return out;
    }
    const int64_t vocab = tok->ne[1];
    const size_t  seq   = input_ids.size();
    out.resize(seq * (size_t)hidden);

    // 1) Embed lookup: copy token_embd column input_ids[p] into out row p.
    //    Read via ggml_backend_tensor_get so it works for any backend (CPU or
    //    GPU) — token_embd data lives in the loader's backend buffer.
    for (size_t p = 0; p < seq; ++p) {
        int32_t t = input_ids[p];
        if (t < 0 || t >= vocab) {
            MT_LOGE("fuse_embeds: input_ids[%zu]=%d out of range [0,%lld)",
                    p, t, (long long)vocab);
            out.clear();
            return out;
        }
        const size_t src_off = (size_t)t * (size_t)hidden * sizeof(float);
        ggml_backend_tensor_get(tok, out.data() + p * (size_t)hidden, src_off,
                                (size_t)hidden * sizeof(float));
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
    // Raw F32 read below assumes GGML_TYPE_F32; token_embd is never promoted to
    // F32, so a non-F32 GGUF would read garbage. Fail loud instead.
    if (tok->type != GGML_TYPE_F32) {
        MT_LOGE("embed_token: token_embd type %d != GGML_TYPE_F32 (F32 GGUF required)",
                (int)tok->type);
        return out;
    }
    const int64_t vocab = tok->ne[1];
    if (t < 0 || t >= vocab) {
        MT_LOGE("embed_token: id %d out of range [0,%lld)", (int)t, (long long)vocab);
        return out;
    }
    out.resize((size_t)hidden);
    const size_t src_off = (size_t)t * (size_t)hidden * sizeof(float);
    ggml_backend_tensor_get(tok, out.data(), src_off, (size_t)hidden * sizeof(float));
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
