#include "audio_adaptor.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>

int main() {
    const char* gguf = std::getenv("MTD_TEST_GGUF");
    const char* base = std::getenv("MTD_TEST_BASELINE");
    if (!gguf || !base) return 77;

    mt::ModelLoader m;
    if (!m.load(gguf)) return 1;
    m.promote_small_f16_to_f32();

    // Dumped encoder output: ne=[1024,1500,1]. The trailing dim of 1 collapses
    // (ggml_n_dims==2), so load_baseline reports the 2D shape [1500,1024]
    // (outer..inner): sh[0]=T=1500, sh[1]=D=1024, feature-fastest enc[t*1024 + d].
    std::vector<float> enc;
    std::vector<int64_t> sh;
    if (!mttest::load_baseline(base, "encoder_hidden", enc, sh)) return 1;
    const int T_full = (int)sh[0];  // 1500
    const int D = (int)sh[1];       // 1024

    // The reference get_audio_features trims each chunk to
    // audio_feature_lengths[chunk]*4 frames BEFORE time-merge.
    std::vector<int32_t> lengths;
    if (!mttest::load_baseline_i32(base, "audio_feature_lengths", lengths)) return 1;
    if (lengths.empty()) { std::fprintf(stderr, "empty audio_feature_lengths\n"); return 1; }
    const int L0 = lengths[0];       // 138
    const int T = L0 * 4;            // 552
    if (T > T_full) { std::fprintf(stderr, "trim T=%d > T_full=%d\n", T, T_full); return 1; }

    // Feature-fastest flat layout: the first T frames are the first T*D floats.
    enc.resize((size_t)T * (size_t)D);

    mt::AudioAdaptor ad(m);
    std::vector<float> got;
    int N = 0, H = 0;
    ad.apply(enc, T, D, got, N, H);
    std::fprintf(stderr, "audio_adaptor: N=%d H=%d\n", N, H);

    bool all_ok = true;

    // Optional finer isolation: compare post-time-merge reshape to `merged`.
    // The reshape groups 4 consecutive frames' features into one 4096 vector,
    // so `merged` == the first N*4096 floats of the (feature-fastest) trimmed enc.
    {
        std::vector<float> mref;
        std::vector<int64_t> msh;
        if (mttest::load_baseline(base, "merged", mref, msh)) {
            std::vector<float> mgot(enc.begin(), enc.begin() + (size_t)N * 4096);
            bool ok = mttest::compare(mgot, mref, "time_merge", 1e-3f, 1e-2f);
            all_ok = all_ok && ok;
        }
    }

    std::vector<float> ref;
    std::vector<int64_t> rsh;
    if (!mttest::load_baseline(base, "audio_embeds", ref, rsh)) return 1;  // [138,1024]
    bool ok = mttest::compare(got, ref, "audio_adaptor", 1e-3f, 1e-2f);
    all_ok = all_ok && ok;

    return all_ok ? 0 : 1;
}
