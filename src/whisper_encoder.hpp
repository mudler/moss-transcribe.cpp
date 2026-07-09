#pragma once
#include <vector>
#include "model_loader.hpp"
#include <ggml.h>
namespace mt {
struct WhisperLayer {
    ggml_tensor *attn_ln_w, *attn_ln_b;
    ggml_tensor *q_w, *q_b, *k_w, *v_w, *v_b, *o_w, *o_b;   // k has NO bias
    ggml_tensor *ffn_ln_w, *ffn_ln_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};
class WhisperEncoder {
public:
    explicit WhisperEncoder(ModelLoader& m);
    void encode(const std::vector<float>& mel, int n_mels, int n_frames,
                std::vector<float>& out, int& out_T, int& out_D) const;
private:
    int d_model_, n_layers_, n_heads_, ffn_, max_src_pos_;
    ggml_tensor *conv1_w_, *conv1_b_, *conv2_w_, *conv2_b_, *pos_embd_, *ln_post_w_, *ln_post_b_;
    std::vector<WhisperLayer> layers_;
};
} // namespace mt
