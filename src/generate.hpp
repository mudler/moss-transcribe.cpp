#ifndef MT_GENERATE_HPP
#define MT_GENERATE_HPP

// Generation helpers for moss-transcribe.cpp.
//
// fuse_embeds: builds the fused input embeddings the Qwen3 decoder consumes.
// It performs the embed lookup (token_embd.weight rows for input_ids) followed
// by the audio injection (masked_scatter): each position whose id equals
// audio_token_id is overwritten, in increasing index order, with the next
// audio_embeds row. This is exactly torch masked_scatter semantics.

#include "model_loader.hpp"

#include <cstdint>
#include <vector>

namespace mt {

// Build the fused input embeddings [hidden x seq] (feature-fastest, token-major
// flat: out[p*hidden + h]).
//
// - input_ids:    seq token ids.
// - audio_embeds: n_audio audio rows, token-major flat (audio_embeds[k*hidden+h]).
// - n_audio:      number of audio rows available (== count of audio_token_id
//                 positions in input_ids for a well-formed input).
// - hidden:       model hidden size.
// - audio_token_id: placeholder token id marking audio positions.
//
// The base embedding of token t is column t of token_embd.weight
// (ne=[hidden,vocab]) i.e. token_embd_data[t*hidden + h]. The k-th audio-token
// position (in increasing index) is overwritten with audio_embeds row k.
std::vector<float> fuse_embeds(ModelLoader& m,
                               const std::vector<int32_t>& input_ids,
                               const std::vector<float>& audio_embeds,
                               int n_audio, int hidden, int audio_token_id);

}  // namespace mt

#endif  // MT_GENERATE_HPP
