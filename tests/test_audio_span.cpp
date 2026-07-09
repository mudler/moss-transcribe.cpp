#include "audio_span.hpp"
#include "tokenizer.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
static int check(const char* gguf, const char* base) {
    mt::ModelLoader m; if (!m.load(gguf)) return 1;
    mt::Tokenizer tok; if (!tok.load(m)) return 1;
    std::vector<int32_t> ref; if (!mttest::load_baseline_i32(base, "input_ids", ref)) return 1;
    // num_audio_tokens = count of audio_token_id in ref
    int naudio = 0; for (int32_t id : ref) if (id == m.config().audio_token_id) ++naudio;
    auto got = mt::build_input_ids(tok, m.config(), m.config().default_prompt, naudio);
    if (got.size() != ref.size()) { std::fprintf(stderr,"len %zu vs %zu\n", got.size(), ref.size()); return 1; }
    for (size_t i=0;i<got.size();++i) if (got[i]!=ref[i]) {
        std::fprintf(stderr,"mismatch at %zu: got %d ref %d\n", i, got[i], ref[i]); return 1; }
    std::printf("ok input_ids bit-exact (%zu tokens, %d audio)\n", got.size(), naudio);
    return 0;
}
int main() {
    const char* gguf = std::getenv("MTD_TEST_GGUF");
    const char* bs = std::getenv("MTD_TEST_BASELINE");         // short
    const char* bl = std::getenv("MTD_TEST_BASELINE_LONG");    // long
    if (!gguf || !bs) return 77;
    if (check(gguf, bs)) return 1;
    if (bl && check(gguf, bl)) return 1;   // long fixture if provided
    return 0;
}
