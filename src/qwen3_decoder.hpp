#ifndef MT_QWEN3_DECODER_HPP
#define MT_QWEN3_DECODER_HPP

// Qwen3-0.6B text decoder for moss-transcribe.cpp.
//
// The transformer stack itself is the vendored `qwen3_layer_forward` (same
// Qwen3 arch as moss-tts: NEOX RoPE base 1e6, per-head q/k RMSNorm before RoPE,
// GQA via mul_mat broadcast, GGML_PREC_F32 scores, SwiGLU). This driver:
//   - builds Qwen3Hparams from mt::Config (NOT from any qwen3.* KV, which the
//     converter did not write),
//   - loads token_embd.weight (tied lm_head) + qwen3.output_norm.weight,
//   - owns a persistent per-layer device-resident K/V cache,
//   - prefills S embedding rows and returns the final-RMSNorm'd hidden for
//     EVERY position (so parity can be checked per-position),
//   - applies the tied lm_head (token_embd.weight @ hidden) on demand.

#include "qwen3.hpp"
#include "model_loader.hpp"
#include "ggml_extend.hpp"   // GgmlCtxPtr
#include "ggml-backend.h"    // ggml_backend_buffer_t

#include <cstdint>
#include <vector>

namespace mt {

class Qwen3Decoder {
public:
    Qwen3Decoder() = default;
    ~Qwen3Decoder();
    Qwen3Decoder(const Qwen3Decoder&) = delete;             // owns a backend buffer
    Qwen3Decoder& operator=(const Qwen3Decoder&) = delete;

    // Build hparams from m.config(), load layers + token_embd + output_norm,
    // and allocate the persistent K/V cache for up to max_seq positions.
    bool load(const ModelLoader& m, int max_seq);

    int hidden() const { return hp_.hidden; }
    const Qwen3Hparams& hparams() const { return hp_; }

    // Prefill S embedding rows (token-major, S*hidden floats: row s = positions
    // s's hidden vector). Resets the cache to position 0. Writes the per-position
    // final-RMSNorm'd hidden states (token-major S*hidden) into *all_hidden.
    bool prefill(const std::vector<float>& embeds, int S, std::vector<float>* all_hidden);

    // Tied lm_head: logits = token_embd.weight @ hidden_row -> [vocab].
    std::vector<float> logits_from_hidden(const std::vector<float>& hidden_row);

    // One decode step at the next position. embed: [hidden]. Returns the
    // final-RMSNorm'd hidden [hidden]; advances past_len().
    std::vector<float> decode_one(const std::vector<float>& embed);

    void reset();
    int  past_len() const { return past_len_; }

private:
    bool run(const std::vector<float>& embeds, int T, std::vector<float>* out_hidden);

    Qwen3Hparams hp_{};
    std::vector<Qwen3Layer> layers_;
    struct ggml_tensor* output_norm_ = nullptr;
    struct ggml_tensor* token_embd_  = nullptr;   // [hidden, vocab] — tied lm_head
    const ModelLoader* m_ = nullptr;
    int max_seq_ = 0, past_len_ = 0;

    // Device-resident per-layer K/V cache: [head_dim, n_kv_heads, max_seq, 1].
    GgmlCtxPtr kv_ctx_;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    std::vector<struct ggml_tensor*> k_cache_, v_cache_;
    std::vector<uint8_t> scratch_;   // reused per-run graph-metadata ctx buffer
};

}  // namespace mt

#endif  // MT_QWEN3_DECODER_HPP
