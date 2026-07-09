#include "transcribe.hpp"

#include "audio_io.hpp"
#include "audio_encoder.hpp"
#include "audio_span.hpp"
#include "tokenizer.hpp"
#include "generate.hpp"
#include "qwen3_decoder.hpp"
#include "common.hpp"

#include <string>
#include <vector>

namespace mt {

static std::string strip(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (unsigned char)s[b] <= ' ') ++b;
    while (e > b && (unsigned char)s[e - 1] <= ' ') --e;
    return s.substr(b, e - b);
}

std::string transcribe_pcm16k(ModelLoader& m, const std::vector<float>& samples,
                              int max_new) {
    const Config& c = m.config();
    const int hidden = c.text_hidden;

    // 2. Audio front end -> audio_embeds [hidden x n_tokens].
    AudioEncoder aenc(m);
    int n_tokens = 0;
    std::vector<float> audio_embeds = aenc.encode(samples, n_tokens, hidden);
    if (audio_embeds.empty() || n_tokens <= 0) {
        MT_LOGE("transcribe_pcm16k: audio encoding failed");
        return {};
    }

    // 3. Tokenizer + input_ids (prompt with time-marker-interleaved audio span).
    Tokenizer tok;
    if (!tok.load(m)) { MT_LOGE("transcribe_pcm16k: tokenizer load failed"); return {}; }
    std::vector<int32_t> input_ids =
        build_input_ids(tok, c, c.default_prompt, n_tokens);
    if (input_ids.empty()) { MT_LOGE("transcribe_pcm16k: build_input_ids empty"); return {}; }

    // 4. Fuse: embed lookup + masked_scatter audio injection.
    std::vector<float> fused =
        fuse_embeds(m, input_ids, audio_embeds, n_tokens, hidden, c.audio_token_id);
    if (fused.empty()) { MT_LOGE("transcribe_pcm16k: fuse_embeds failed"); return {}; }

    // 5. Greedy generate.
    const int seq = (int)input_ids.size();
    Qwen3Decoder dec;
    if (!dec.load(m, seq + max_new + 16)) { MT_LOGE("transcribe_pcm16k: decoder load failed"); return {}; }
    std::vector<int32_t> new_ids =
        greedy_generate(dec, m, fused, seq, max_new, c.eos_token_id);
    if (new_ids.empty()) { MT_LOGE("transcribe_pcm16k: no tokens generated"); return {}; }

    // 6. Decode (skips special tokens, dropping the trailing EOS) + strip.
    std::string text = tok.decode(new_ids);
    return strip(text);
}

std::string transcribe_wav(ModelLoader& m, const std::string& wav_path, int max_new) {
    // 1. Load + resample audio to 16 kHz mono, then run the shared PCM core.
    Audio a;
    if (!load_audio_16k_mono(wav_path, a)) {
        MT_LOGE("transcribe_wav: failed to load %s", wav_path.c_str());
        return {};
    }
    return transcribe_pcm16k(m, a.samples, max_new);
}

}  // namespace mt
