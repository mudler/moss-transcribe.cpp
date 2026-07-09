#ifndef MT_AUDIO_ENCODER_HPP
#define MT_AUDIO_ENCODER_HPP

// Audio front-end orchestration (get_audio_features equivalent).
//
// Splits 16 kHz mono samples into 30 s chunks. For each chunk it computes the
// true token length from the UNPADDED sample count, pads the chunk to
// feat_n_samples, runs mel -> WhisperEncoder, and trims the padded encoder
// frames to token_len*4. The trimmed encoder frames from ALL chunks are
// concatenated along time, and the AudioAdaptor is run ONCE on the whole
// concatenation (time-merge by audio_merge_size, then MLP) -> audio_embeds
// [hidden x Sum(token_len)]. This matches the reference get_audio_features
// which concatenates trimmed encoder frames FIRST, then merges+projects.

#include "model_loader.hpp"

#include <vector>

namespace mt {

class AudioEncoder {
public:
    explicit AudioEncoder(ModelLoader& m);

    // Encode 16 kHz mono samples to audio embeds, token-major flat
    // (out[k*hidden + h]). n_tokens is set to Sum(token_len) over all chunks.
    // `hidden` is the expected hidden size (adaptor output H); pass -1 to skip
    // the check.
    std::vector<float> encode(const std::vector<float>& samples16k,
                              int& n_tokens, int hidden);

private:
    ModelLoader& m_;
};

}  // namespace mt

#endif  // MT_AUDIO_ENCODER_HPP
