// Task 14 decisive gates.
// GATE A: greedy loop from dumped fused_embeds reproduces reference generated_ids
//         (isolates decode_one KV continuation, embed lookup, argmax tie-break).
// GATE B (THE GOAL): full wav -> transcript text == baseline.text.
#include "transcribe.hpp"
#include "generate.hpp"
#include "audio_encoder.hpp"
#include "qwen3_decoder.hpp"
#include "audio_span.hpp"
#include "tokenizer.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
int main() {
    const char* gguf=std::getenv("MTD_TEST_GGUF"); const char* base=std::getenv("MTD_TEST_BASELINE");
    if(!gguf||!base) return 77;
    mt::ModelLoader m; if(!m.load(gguf)) return 1; m.promote_small_f16_to_f32();
    // GATE A: greedy generated ids from dumped fused_embeds match reference generated_ids
    std::vector<float> fused; std::vector<int64_t> sh; mttest::load_baseline(base,"fused_embeds",fused,sh);
    std::vector<int32_t> ref_gen; mttest::load_baseline_i32(base,"generated_ids",ref_gen);
    std::vector<int32_t> ref_ids; mttest::load_baseline_i32(base,"input_ids",ref_ids);
    // fused_embeds ne=[hidden, seq]; load_baseline reports outer..inner so
    // sh={seq, hidden} (token-major, same convention as Task 12): seq=sh[0].
    // Pass a GENEROUS max_new (120 > ~71) so a bit-exact match also proves the
    // natural EOS-stop (not truncation).
    mt::Qwen3Decoder dec; dec.load(m, (int)ref_gen.size()+16);
    auto newids = mt::greedy_generate(dec, m, fused, (int)sh[0], 120, m.config().eos_token_id);
    std::vector<int32_t> ref_new(ref_gen.begin()+(long)ref_ids.size(), ref_gen.end());
    if (newids.size()!=ref_new.size()) { std::fprintf(stderr,"gen len %zu vs %zu\n",newids.size(),ref_new.size()); return 1; }
    for (size_t i=0;i<newids.size();++i) if (newids[i]!=ref_new[i]) { std::fprintf(stderr,"gen mismatch @%zu %d vs %d\n",i,newids[i],ref_new[i]); return 1; }
    std::printf("GATE A OK: %zu generated_ids match\n", newids.size());
    // GATE B (E2E vs UPSTREAM): full wav->transcript text == baseline.text
    std::string txt = mt::transcribe_wav(m, "tests/fixtures/short.wav", 300);
    std::string ref; mttest::load_kv_str(base,"baseline.text",ref);
    if (txt != ref) { std::fprintf(stderr,"E2E mismatch:\n got:[%s]\n ref:[%s]\n", txt.c_str(), ref.c_str()); return 1; }
    std::printf("E2E OK: %s\n", txt.c_str());
    return 0;
}
