#ifndef MT_AUDIO_SPAN_HPP
#define MT_AUDIO_SPAN_HPP

// Audio-span builder + input_ids assembly (M4b).
//
// Pure integer logic (no ggml). Ports the reference
// `processing._audio_span_ids` (time-marker interleaver) and the
// `_expand_audio_token` chat-template assembly.

#include "config.hpp"
#include "tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mt {

// Exact port of `processing._audio_span_ids`. Emits `audio_seq_len` audio
// tokens, interleaved with digit tokens marking elapsed whole seconds every
// `time_marker_every_seconds`. Matches Python integer semantics exactly.
std::vector<int32_t> audio_span_ids(const Config& cfg, const Tokenizer& tok,
                                    int audio_seq_len);

// Render the fixed chat-template prompt, expand the single `<|audio_pad|>`
// into the time-marker-interleaved audio span, and return the full token
// sequence (the reference `_expand_audio_token`).
std::vector<int32_t> build_input_ids(const Tokenizer& tok, const Config& cfg,
                                     const std::string& user_prompt,
                                     int num_audio_tokens);

}  // namespace mt

#endif  // MT_AUDIO_SPAN_HPP
