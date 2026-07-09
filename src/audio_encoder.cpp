#include "audio_encoder.hpp"

#include "mel.hpp"
#include "whisper_encoder.hpp"
#include "audio_adaptor.hpp"
#include "common.hpp"   // MT_LOGE / MT_LOGW

#include <algorithm>
#include <cstddef>
#include <vector>

namespace mt {

AudioEncoder::AudioEncoder(ModelLoader& m) : m_(m) {}

std::vector<float> AudioEncoder::encode(const std::vector<float>& samples16k,
                                        int& n_tokens, int hidden) {
    n_tokens = 0;
    std::vector<float> out;

    const Config& c = m_.config();
    const int chunk_samples = c.feat_n_samples;              // 480000 (30 s)
    // stride = feat_hop * WHISPER_ENCODER_STRIDE(2) * audio_merge_size.
    const int WHISPER_ENCODER_STRIDE = 2;
    const int stride = c.feat_hop * WHISPER_ENCODER_STRIDE * c.audio_merge_size;  // 1280
    if (chunk_samples <= 0 || stride <= 0) {
        MT_LOGE("AudioEncoder: bad config (chunk=%d stride=%d)", chunk_samples, stride);
        return out;
    }
    if (samples16k.empty()) {
        MT_LOGE("AudioEncoder: empty input samples");
        return out;
    }

    WhisperMel     mel(m_);
    WhisperEncoder enc(m_);

    // Accumulate trimmed encoder frames from every chunk, feature-fastest
    // (concat[t*eD + d]), then run the adaptor once on the whole thing.
    std::vector<float> concat;
    int eD = 0;
    int total_token_len = 0;

    const size_t total = samples16k.size();
    for (size_t off = 0; off < total; off += (size_t)chunk_samples) {
        const size_t unpadded = std::min((size_t)chunk_samples, total - off);
        const int token_len = (int)((unpadded - 1) / (size_t)stride) + 1;
        total_token_len += token_len;

        // Pad chunk to feat_n_samples.
        std::vector<float> chunk(samples16k.begin() + (long)off,
                                 samples16k.begin() + (long)(off + unpadded));
        chunk.resize((size_t)chunk_samples, 0.0f);

        std::vector<float> feat; int n_mels = 0, T = 0;
        mel.compute(chunk, feat, n_mels, T);

        std::vector<float> eh; int eT = 0, cD = 0;
        enc.encode(feat, n_mels, T, eh, eT, cD);
        if (eD == 0) eD = cD;
        else if (cD != eD) { MT_LOGE("AudioEncoder: eD mismatch %d vs %d", cD, eD); return out; }

        const int Ttrim = token_len * c.audio_merge_size;  // token_len * 4
        if (Ttrim > eT) { MT_LOGE("AudioEncoder: trim %d > eT=%d", Ttrim, eT); return out; }

        concat.insert(concat.end(), eh.begin(),
                      eh.begin() + (size_t)Ttrim * (size_t)eD);
    }

    if (concat.empty() || eD <= 0) { MT_LOGE("AudioEncoder: no encoder frames"); return out; }

    const int concatT = (int)(concat.size() / (size_t)eD);

    AudioAdaptor ad(m_);
    int N = 0, H = 0;
    ad.apply(concat, concatT, eD, out, N, H);
    if (N != total_token_len) {
        MT_LOGW("AudioEncoder: adaptor N=%d != Sum(token_len)=%d", N, total_token_len);
    }
    if (hidden > 0 && H != hidden) {
        MT_LOGE("AudioEncoder: adaptor H=%d != expected hidden=%d", H, hidden);
        out.clear();
        return out;
    }
    n_tokens = N;
    return out;
}

}  // namespace mt
