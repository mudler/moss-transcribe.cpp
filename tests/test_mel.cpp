#include "audio_io.hpp"
#include "mel.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
int main() {
    const char* gguf = std::getenv("MTD_TEST_GGUF");
    const char* base = std::getenv("MTD_TEST_BASELINE");
    if (!gguf || !base) return 77;                 // skip if unset
    mt::ModelLoader m;
    if (!m.load(gguf)) { std::fprintf(stderr, "load gguf failed\n"); return 1; }
    mt::Audio a;
    if (!mt::load_audio_16k_mono("tests/fixtures/short.wav", a)) return 1;
    // short.wav is < 30s -> pad/trim to n_samples (single chunk), matching processor
    a.samples.resize(m.config().feat_n_samples, 0.0f);
    mt::WhisperMel mel(m);
    std::vector<float> got; int n_mels=0, T=0;
    mel.compute(a.samples, got, n_mels, T);         // [80, 3000] row-major
    std::vector<float> ref; std::vector<int64_t> shape;
    if (!mttest::load_baseline(base, "input_features", ref, shape)) return 1;
    // ref shape [1,80,3000]; drop the batch dim -> compare [80,3000]
    bool ok = mttest::compare(got, ref, "mel", /*atol*/1e-3f, /*rtol*/1e-2f);
    return ok ? 0 : 1;
}
