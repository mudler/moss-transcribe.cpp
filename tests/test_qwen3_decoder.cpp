// Qwen3 decoder parity test. Feeds the dumped `fused_embeds` (the post
// audio-injection input embeddings) straight into the decoder, so this isolates
// the transformer stack + tied lm_head from the front end.
//
// Layout note: load_baseline reports shape outer..inner, so for a torch
// (seq, hidden) dump it returns sh=[seq, hidden] and the flat data is
// token-major (row s = position s's hidden vector, s*hidden + h). Verified
// empirically against the fixture (sh=[227,1024]); the decoder consumes/produces
// the same token-major layout, so `hid` compares elementwise against `lm_hidden`.
#include "qwen3_decoder.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>

int main() {
    const char* gguf = std::getenv("MTD_TEST_GGUF");
    const char* base = std::getenv("MTD_TEST_BASELINE");
    if (!gguf || !base) return 77;

    mt::ModelLoader m;
    if (!m.load(gguf)) return 1;
    m.promote_small_f16_to_f32();

    std::vector<float> fused; std::vector<int64_t> sh;
    if (!mttest::load_baseline(base, "fused_embeds", fused, sh)) return 1;  // [seq, hidden]
    const int seq = (int)sh[0], H = (int)sh[1];

    mt::Qwen3Decoder dec;
    if (!dec.load(m, seq + 512)) return 1;

    std::vector<float> hid;
    if (!dec.prefill(fused, seq, &hid)) return 1;                           // [seq, hidden]

    std::vector<float> ref_h; std::vector<int64_t> hsh;
    if (!mttest::load_baseline(base, "lm_hidden", ref_h, hsh)) return 1;
    bool ok = mttest::compare(hid, ref_h, "lm_hidden", 5e-2f, 5e-2f);

    // argmax of last-position logits must match the reference greedy first token.
    std::vector<float> last(hid.end() - H, hid.end());
    auto logits = dec.logits_from_hidden(last);
    if (logits.empty()) { std::fprintf(stderr, "logits_from_hidden failed\n"); return 1; }
    int am = 0;
    for (int i = 1; i < (int)logits.size(); ++i) if (logits[i] > logits[am]) am = i;

    std::vector<float> ref_logits; std::vector<int64_t> lsh;
    if (!mttest::load_baseline(base, "prompt_logits", ref_logits, lsh)) return 1;  // [seq, vocab]
    const int V = (int)lsh[1];
    int ram = 0; const float* rl = &ref_logits[(size_t)(seq - 1) * V];
    for (int i = 1; i < V; ++i) if (rl[i] > rl[ram]) ram = i;

    if (am != ram) { std::fprintf(stderr, "argmax %d vs ref %d\n", am, ram); ok = false; }
    std::printf("qwen3 lm_hidden + last-pos argmax (%d==%d)\n", am, ram);
    return ok ? 0 : 1;
}
