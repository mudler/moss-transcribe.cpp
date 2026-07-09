// Phase 1 end-to-end front-end parity: chain audio_io -> mel -> whisper_encoder
// -> audio_adaptor from a raw wav and confirm it reproduces the dumped
// `audio_embeds`. This composes the already-passing Phase 1 components plus the
// per-chunk encoder trim that the C++ get_audio_features equivalent performs.
//
// ggml ne=[D,T] (D contiguous) has the same byte order as torch [T,D] row-major
// (D contiguous), so the flat buffers compare directly.
#include "audio_io.hpp"
#include "mel.hpp"
#include "whisper_encoder.hpp"
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
    if (!m.load(gguf)) { std::fprintf(stderr, "load gguf failed\n"); return 1; }
    m.promote_small_f16_to_f32();
    const mt::Config& c = m.config();

    // 1. Load raw audio (jfk short.wav ~11 s -> ~176000 samples at 16 kHz).
    mt::Audio a;
    if (!mt::load_audio_16k_mono("tests/fixtures/short.wav", a)) {
        std::fprintf(stderr, "load_audio failed\n"); return 1;
    }
    const size_t n_unpadded = a.samples.size();

    // 2. Compute the true token length from the UNPADDED sample count -- this is
    //    the reference get_audio_features logic, not a baseline lookup.
    //    stride = feat_hop * WHISPER_ENCODER_STRIDE(2) * audio_merge_size.
    const int WHISPER_ENCODER_STRIDE = 2;
    const int stride = c.feat_hop * WHISPER_ENCODER_STRIDE * c.audio_merge_size; // 160*2*4=1280
    const int token_len = (int)((n_unpadded - 1) / (size_t)stride) + 1;          // 138 for jfk
    std::fprintf(stderr, "e2e: n_unpadded=%zu stride=%d token_len=%d\n",
                 n_unpadded, stride, token_len);
    if (token_len != 138) {
        std::fprintf(stderr, "token_len=%d != expected 138\n", token_len);
        return 1;
    }

    // 3. Pad/trim to a single 30 s chunk (feat_n_samples = 480000).
    a.samples.resize(c.feat_n_samples, 0.0f);

    // 4. mel -> [80, 3000].
    mt::WhisperMel mel(m);
    std::vector<float> feat; int n_mels = 0, T = 0;
    mel.compute(a.samples, feat, n_mels, T);

    // 5. encoder -> [1024, 1500] (feature-fastest eh[t*1024 + d]).
    mt::WhisperEncoder enc(m);
    std::vector<float> eh; int eT = 0, eD = 0;
    enc.encode(feat, n_mels, T, eh, eT, eD);

    // 6. Trim the padded encoder output to token_len*4 frames before the adaptor.
    //    Feature-fastest layout: the first token_len*4 frames are the first
    //    token_len*4*eD floats.
    const int Ttrim = token_len * 4; // 552
    if (Ttrim > eT) { std::fprintf(stderr, "trim %d > eT=%d\n", Ttrim, eT); return 1; }
    eh.resize((size_t)Ttrim * (size_t)eD);
    eT = Ttrim;

    // 7. adaptor -> N = token_len = 138.
    mt::AudioAdaptor ad(m);
    std::vector<float> emb; int N = 0, H = 0;
    ad.apply(eh, eT, eD, emb, N, H);
    std::fprintf(stderr, "e2e: N=%d H=%d\n", N, H);
    if (N != 138) { std::fprintf(stderr, "N=%d != expected 138\n", N); return 1; }

    // 8. Compare to the dumped audio_embeds (138x1024). Looser tolerance to
    //    absorb accumulation across mel + encoder + adaptor; cos still >= 0.99999.
    std::vector<float> ref; std::vector<int64_t> rsh;
    if (!mttest::load_baseline(base, "audio_embeds", ref, rsh)) return 1;
    bool ok = mttest::compare(emb, ref, "audio_path_e2e", /*atol*/5e-2f, /*rtol*/5e-2f);
    return ok ? 0 : 1;
}
