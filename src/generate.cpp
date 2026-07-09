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

}  // namespace mt
