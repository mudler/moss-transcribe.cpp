#pragma once
#include <vector>
#include "model_loader.hpp"
#include <ggml.h>

namespace mt {

// Audio adaptor: 4x time-merge + VQAdaptor
// (Linear 4096->1024 +b -> SiLU -> Linear 1024->1024 +b -> LayerNorm(1024, eps=1e-6) +b).
// Input: concatenated + already-trimmed encoder output [D=1024, T] (feature fastest).
// Output: audio embeds [H=1024, N=T/merge].
class AudioAdaptor {
public:
    explicit AudioAdaptor(ModelLoader& m);
    void apply(const std::vector<float>& enc, int T, int D,
               std::vector<float>& out, int& N, int& H) const;

private:
    int merge_, in_dim_, hidden_;
    float eps_;
    ggml_tensor *fc1_w_, *fc1_b_, *fc2_w_, *fc2_b_, *ln_w_, *ln_b_;
};

}  // namespace mt
