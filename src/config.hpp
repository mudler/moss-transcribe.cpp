#pragma once
#include <string>
#include <vector>
namespace mt {
struct Config {
    std::string arch;
    // text (qwen3)
    int text_vocab, text_hidden, text_ffn, text_layers, text_heads, text_kv_heads, text_head_dim;
    float text_rms_eps, text_rope_theta; int text_max_pos; bool text_tied;
    // audio (whisper)
    int audio_mel_bins, audio_d_model, audio_layers, audio_heads, audio_ffn, audio_max_src_pos;
    std::string audio_act; bool audio_scale_embed;
    // bridge
    int audio_token_id, audio_merge_size, adaptor_input_dim; float adaptor_norm_eps;
    // mel
    int feat_sr, feat_n_fft, feat_hop, feat_size, feat_n_samples, feat_nb_max_frames; float feat_dither;
    // audio-span
    float audio_tokens_per_second; int time_marker_every_seconds; bool enable_time_marker;
    // generation
    int eos_token_id, pad_token_id, default_max_new_tokens; std::string default_prompt;
    std::vector<int32_t> digit_token_ids; // 10 ids for "0".."9"
};
} // namespace mt
