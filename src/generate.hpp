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
#include "qwen3_decoder.hpp"

#include <cstdint>
#include <vector>

namespace mt {

// Fetch the embedding of token id `t` = column t of token_embd.weight
// (ne=[hidden,vocab]) -> [hidden]. Returns empty on error.
std::vector<float> embed_token(ModelLoader& m, int32_t t, int hidden);

// Greedy autoregressive generation matching torch/HF greedy decoding exactly.
//
// - Prefill `fused` (seq token-major embedding rows, seq*hidden floats).
// - Argmax the last-position logits (FIRST max index on ties) -> next token,
//   append it. If it is `eos`, stop (the returned ids INCLUDE the trailing EOS).
//   Else if we have produced max_new tokens, stop. Otherwise embed the new
//   token, decode_one (advancing the KV cache), argmax again.
//
// Returns the generated token ids (INCLUDING the trailing EOS when hit).
std::vector<int32_t> greedy_generate(Qwen3Decoder& dec, ModelLoader& m,
                                     const std::vector<float>& fused, int seq,
                                     int max_new, int eos);

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
