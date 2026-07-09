#include "generate.hpp"
#include "parity.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

// Fusion parity: embed lookup + masked_scatter audio injection vs the dumped
// baseline `fused_embeds`. Should be near bit-exact (it is a copy).
static int check(const char* gguf, const char* base) {
    mt::ModelLoader m;
    if (!m.load(gguf)) return 1;
    m.promote_small_f16_to_f32();

    std::vector<int32_t> input_ids;
    if (!mttest::load_baseline_i32(base, "input_ids", input_ids)) return 1;

    std::vector<float>   audio_embeds;
    std::vector<int64_t> ashape;
    if (!mttest::load_baseline(base, "audio_embeds", audio_embeds, ashape)) return 1;

    std::vector<float>   ref;
    std::vector<int64_t> fshape;
    if (!mttest::load_baseline(base, "fused_embeds", ref, fshape)) return 1;

    // audio_embeds ne=[hidden, n_audio]; load_baseline reports shape outer..inner
    // -> shape = {n_audio, hidden}. fused_embeds ne=[hidden, seq].
    const int hidden  = (int)ashape.back();
    const int n_audio = (int)ashape.front();
    const int audio_token_id = m.config().audio_token_id;

    std::printf("hidden=%d n_audio=%d seq=%zu audio_token_id=%d\n",
                hidden, n_audio, input_ids.size(), audio_token_id);

    auto got = mt::fuse_embeds(m, input_ids, audio_embeds, n_audio, hidden,
                               audio_token_id);
    if (got.empty()) { std::fprintf(stderr, "fuse_embeds returned empty\n"); return 1; }

    if (!mttest::compare(got, ref, "fused_embeds", 1e-4f, 1e-3f)) return 1;
    return 0;
}

int main() {
    const char* gguf = std::getenv("MTD_TEST_GGUF");
    const char* bs   = std::getenv("MTD_TEST_BASELINE");
    if (!gguf || !bs) return 77;
    return check(gguf, bs);
}
