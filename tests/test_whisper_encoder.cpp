#include "whisper_encoder.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
// ggml ne=[D,T] column-major storage has D contiguous, identical byte order to
// torch [T,D] row-major with D contiguous -> the flat buffers line up directly.
int main() {
    const char* gguf = std::getenv("MTD_TEST_GGUF");
    const char* base = std::getenv("MTD_TEST_BASELINE");
    if (!gguf || !base) return 77;
    mt::ModelLoader m;
    if (!m.load(gguf)) return 1;
    m.promote_small_f16_to_f32();
    // feed the dumped input_features (first chunk), not our own mel -> isolate the encoder
    std::vector<float> feat; std::vector<int64_t> sh;
    if (!mttest::load_baseline(base, "input_features", feat, sh)) return 1; // [80,3000] (leading batch dim collapsed)
    // Take the last two dims as (n_mels, n_frames); robust whether the dumped
    // tensor is stored 2-D [80,3000] or 3-D [1,80,3000].
    if (sh.size() < 2) return 1;
    const int n_mels = (int)sh[sh.size()-2], n_frames = (int)sh[sh.size()-1];
    std::vector<float> chunk0(feat.begin(), feat.begin() + (size_t)n_mels*n_frames);
    mt::WhisperEncoder enc(m);
    std::vector<float> got; int T=0, D=0;
    enc.encode(chunk0, n_mels, n_frames, got, T, D);        // [1024,1500]
    std::vector<float> ref; std::vector<int64_t> rsh;
    if (!mttest::load_baseline(base, "encoder_hidden", ref, rsh)) return 1; // [1,1500,1024]
    // ref is [T=1500, D=1024] row-major (D fastest); got is [D=1024,T=1500] (D fastest per column)
    // both have D fastest -> memory-compatible; compare directly
    bool ok = mttest::compare(got, ref, "whisper_encoder", /*atol*/5e-2f, /*rtol*/5e-2f);
    return ok ? 0 : 1;
}
