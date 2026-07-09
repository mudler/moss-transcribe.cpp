#include "audio_span.hpp"

#include <string>

namespace mt {

// Exact port of `processing._audio_span_ids`. See task-11 notes.
std::vector<int32_t> audio_span_ids(const Config& cfg, const Tokenizer& /*tok*/,
                                    int audio_seq_len) {
    const int32_t audio_id = (int32_t)cfg.audio_token_id;
    const int every = cfg.time_marker_every_seconds;

    std::vector<int32_t> out;
    if (!cfg.enable_time_marker || audio_seq_len <= 0 || every <= 0) {
        if (audio_seq_len > 0) out.assign((size_t)audio_seq_len, audio_id);
        return out;
    }

    // int(12.5 * 5) = 62  (truncation toward zero, not rounding)
    const int tokens_per_marker = (int)(cfg.audio_tokens_per_second * (double)every);
    if (tokens_per_marker <= 0) {
        out.assign((size_t)audio_seq_len, audio_id);
        return out;
    }

    // float division (Python true division), then int() truncation
    const double duration = (double)audio_seq_len / (double)cfg.audio_tokens_per_second;
    const int duration_i = (int)duration;

    int consumed = 0;
    for (int sec = every; sec <= duration_i; sec += every) {
        const int pos = (sec / every) * tokens_per_marker;  // integer division sec/every
        const int seg = pos - consumed;
        if (seg > 0) {
            out.insert(out.end(), (size_t)seg, audio_id);
            consumed += seg;
        }
        const std::string s = std::to_string(sec);
        for (char ch : s) {
            out.push_back(cfg.digit_token_ids[(size_t)(ch - '0')]);
        }
    }

    const int remainder = audio_seq_len - consumed;
    if (remainder > 0) {
        out.insert(out.end(), (size_t)remainder, audio_id);
    }
    return out;
}

std::vector<int32_t> build_input_ids(const Tokenizer& tok, const Config& cfg,
                                     const std::string& user_prompt,
                                     int num_audio_tokens) {
    // Fixed chat-template prompt (real newlines). {prompt} = user_prompt.
    const std::string prompt =
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|>\n" +
        user_prompt +
        "<|im_end|>\n"
        "<|im_start|>assistant\n";

    // Split on the single literal <|audio_pad|>.
    const std::string pad = "<|audio_pad|>";
    const size_t at = prompt.find(pad);
    const std::string before = prompt.substr(0, at);
    const std::string after = prompt.substr(at + pad.size());

    std::vector<int32_t> before_ids = tok.encode(before);
    std::vector<int32_t> after_ids = tok.encode(after);
    std::vector<int32_t> span = audio_span_ids(cfg, tok, num_audio_tokens);

    std::vector<int32_t> out;
    out.reserve(before_ids.size() + span.size() + after_ids.size());
    out.insert(out.end(), before_ids.begin(), before_ids.end());
    out.insert(out.end(), span.begin(), span.end());
    out.insert(out.end(), after_ids.begin(), after_ids.end());
    return out;
}

}  // namespace mt
