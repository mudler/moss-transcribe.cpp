#include "tokenizer.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
int main() {
    const char* gguf = std::getenv("MTD_TEST_GGUF");
    const char* base = std::getenv("MTD_TEST_BASELINE");
    if (!gguf || !base) return 77;
    mt::ModelLoader m; if (!m.load(gguf)) return 1;
    mt::Tokenizer tok; if (!tok.load(m)) { std::fprintf(stderr,"tok load failed\n"); return 1; }
    // 1) special ids
    if (tok.eos_id() != 151645) { std::fprintf(stderr,"eos %d\n", tok.eos_id()); return 1; }
    if (tok.token_to_id("<|audio_pad|>") != 151671) return 1;
    // 2) round-trip on ASCII + CJK
    for (const char* s : {"Hello, world.", "ask what you can do", "请将音频转写为文本"}) {
        auto ids = tok.encode(s);
        if (tok.decode(ids) != s) { std::fprintf(stderr,"roundtrip fail: %s -> %s\n", s, tok.decode(ids).c_str()); return 1; }
    }
    // 3) decode the reference generated tokens (after the prompt) == baseline.text
    std::vector<int32_t> gen; if (!mttest::load_baseline_i32(base, "generated_ids", gen)) return 1;
    std::vector<int32_t> ids_prompt; if (!mttest::load_baseline_i32(base, "input_ids", ids_prompt)) return 1;
    std::vector<int32_t> newtok(gen.begin() + (long)ids_prompt.size(), gen.end());
    std::string txt = tok.decode(newtok);
    // strip leading/trailing whitespace
    auto a = txt.find_first_not_of(" \t\n"); auto b = txt.find_last_not_of(" \t\n");
    txt = (a==std::string::npos) ? "" : txt.substr(a, b-a+1);
    std::string ref; if (!mttest::load_kv_str(base, "baseline.text", ref)) return 1;
    if (txt != ref) { std::fprintf(stderr,"decode mismatch:\n got: %s\n ref: %s\n", txt.c_str(), ref.c_str()); return 1; }
    std::printf("ok tokenizer: eos/audio ids + roundtrip + decode==baseline.text\n");
    return 0;
}
