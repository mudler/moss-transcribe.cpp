# moss-transcribe.cpp — Phase 2: Tokenizer, Decode & E2E Transcription (M4–M5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Complete the model to a **working end-to-end transcription from a fixture that matches the upstream PyTorch output**: Qwen2 BPE tokenizer → audio-span builder with time markers (bit-exact `input_ids`) → Qwen3-0.6B decoder (parity vs reference logits) → masked_scatter fusion → greedy generate → `transcribe` CLI whose text equals `baseline.text`.

**Architecture:** Builds on Phase 1's proven audio front end (wav→`audio_embeds`, all cos=1.0). Vendors the Qwen3 decoder graph + Qwen2 byte-BPE tokenizer from `moss-tts.cpp`, adapted to read `mt::Config` and the `qwen3.blk.{i}.*.weight` / `token_embd.weight` / `qwen3.output_norm.weight` tensors already in the GGUF. Decode is autoregressive greedy over the fused audio+text embedding stream; the transcript (with `[start][Sxx]text[end]`) is emitted as plain text tokens.

**Tech Stack:** C++17, ggml (submodule, already pinned). No new Python needed (baselines already dumped in Phase 1).

## Global Constraints

- Namespace `mt::`; GGUF KV prefix `mtd.`; C-API prefix `moss_transcribe_capi_`; test env `MTD_TEST_GGUF`, `MTD_TEST_BASELINE`. Parity F32/CPU.
- Text/Qwen3 constants (read from GGUF via `mt::Config`, never hardcode): hidden 1024, ffn 3072, 28 layers, 16 Q heads, 8 KV heads, **head_dim 128** (independent of hidden/n_heads), rms_eps 1e-6, **rope_theta 1e6**, vocab 151936, tied lm_head. RoPE = **NEOX (half-split)**. q_norm/k_norm = per-head RMSNorm over head_dim applied **BEFORE RoPE** using rms_eps. GQA via `ggml_mul_mat` broadcast (n_heads multiple of n_kv_heads). `ggml_mul_mat_set_prec(scores, GGML_PREC_F32)`. Qwen3 attention has **no biases**. SwiGLU MLP.
- Decoder tensor names (already in GGUF): `qwen3.blk.{i}.{attn_norm,attn_q,attn_k,attn_v,attn_o,attn_q_norm,attn_k_norm,ffn_norm,ffn_gate,ffn_up,ffn_down}.weight`, `qwen3.output_norm.weight`, `token_embd.weight`. **lm_head is tied** → logits = `token_embd.weight @ hidden`. There is NO separate `lm_head` tensor (the checkpoint ships none — the reference requires `tie_weights()`).
- Tokenizer keys in GGUF: `tokenizer.tokens`, `tokenizer.token_type`, `tokenizer.merges`, `tokenizer.special_tokens_ids`, `tokenizer.special_tokens_text`, `tokenizer.bos_id`(-1), `tokenizer.eos_id`(151645), `tokenizer.pad_id`(151643). audio ids: `audio_pad`=151671; `audio_start`/`audio_end` are special tokens. `mtd.digit_token_ids` = 10 ids for "0".."9".
- Audio-span (read from `mt::Config`): `audio_tokens_per_second`=12.5, **`time_marker_every_seconds`=5** (the SHIPPED value — NOT 2), `enable_time_marker`=true, `audio_token_id`=151671, `audio_merge_size`=4.
- Prompt (fixed): `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n`, `{prompt}` = `mtd.default_prompt`. The single `<|audio_pad|>` is expanded by the audio-span builder.
- Generation: greedy (argmax), EOS = 151645, default max_new_tokens 5120 (test uses fewer). Output post-processing = `text.strip()`.
- Trimming/chunking (from Phase 1): audio split into 30s (480000-sample) chunks; per-chunk `token_len=(unpadded_chunk_samples-1)//1280+1` (stride=hop160·2·merge4=1280); each chunk's encoder output trimmed to `token_len·4` frames, all chunks concatenated, then merged+adaptor'd.
- Gold baselines (Phase 1, on disk): `tests/fixtures/baseline_short.gguf` (jfk 11s, 1 chunk, 138 audio tokens, input_ids len 227, generated_ids len 298; `baseline.text` = correct JFK transcript) and `baseline_long.gguf` (>30s, 2 chunks, 550 audio tokens). Tensors: `input_ids`, `fused_embeds` [1024,227], `lm_hidden` [1024,227], `prompt_logits` [151936,227], `generated_ids`, `audio_embeds` [1024,138].
- DRY, YAGNI, TDD, frequent commits. Do NOT loosen parity tolerances to force a pass.

## File Structure (Phase 2)

```
src/tokenizer.{hpp,cpp}          # Qwen2 byte-BPE (vendored+adapted from moss-tts de_tokenizer)
src/audio_span.{hpp,cpp}         # chat-template render + time-marker interleaver -> input_ids (pure int)
src/qwen3.{hpp,cpp}              # vendored qwen3 layer-forward + loader (reads qwen3.blk.{i}.*)
src/qwen3_decoder.{hpp,cpp}      # our driver: KV cache, prefill, decode_one, tied lm_head, final RMSNorm
src/audio_encoder.{hpp,cpp}      # orchestration: chunk audio -> mel+encoder+adaptor per chunk -> trim -> concat -> audio_embeds
src/generate.{hpp,cpp}           # embed lookup + masked_scatter fusion + greedy loop -> token ids
src/transcribe.{hpp,cpp}         # top-level: wav -> audio_embeds + input_ids -> generate -> decode -> text
src/cli.cpp                      # add `transcribe` subcommand
tests/test_tokenizer.cpp, test_audio_span.cpp, test_qwen3_decoder.cpp,
tests/test_fusion.cpp, test_generate_e2e.cpp
```

---

### Task 10: Qwen2 byte-BPE tokenizer (M4a)

**Files:** Create `src/tokenizer.{hpp,cpp}`, `tests/test_tokenizer.cpp`; modify `CMakeLists.txt`, `tests/CMakeLists.txt`.

**Interfaces:**
- Produces: `class mt::Tokenizer { bool load(const mt::ModelLoader&); std::vector<int32_t> encode(const std::string&) const; std::string decode(const std::vector<int32_t>&) const; int32_t eos_id()/pad_id() const; int32_t token_to_id(const std::string&) const; size_t vocab_size() const; };`

- [ ] **Step 1: Vendor `de_tokenizer.{hpp,cpp}` → `tokenizer.{hpp,cpp}`**

```bash
cp /home/mudler/_git/moss-tts.cpp/src/de_tokenizer.hpp src/tokenizer.hpp
cp /home/mudler/_git/moss-tts.cpp/src/de_tokenizer.cpp src/tokenizer.cpp
```
Adapt: class `DeTokenizer` → `mt::Tokenizer`; it reads GGUF keys `tokenizer.tokens`, `tokenizer.token_type`, `tokenizer.merges`, `tokenizer.special_tokens_ids`, `tokenizer.special_tokens_text`, `tokenizer.bos_id/eos_id/pad_id` — these are exactly what the Phase-1 converter wrote, so `load(const mt::ModelLoader&)` works against `models/moss-transcribe-f32.gguf`. Keep the byte-level BPE core and the Qwen2 pre-tokenizer regex approximation intact. Add `token_to_id(const std::string&)` (lookup in the token→id map) if not present.

- [ ] **Step 2: Write failing test `tests/test_tokenizer.cpp`**

```cpp
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
```
Register `mt_add_test(test_tokenizer)` + `LABELS "model"`.

- [ ] **Step 3: RED** — `cmake --build build -j` fails to link (tokenizer.cpp not in MT_SRC yet). Add `src/tokenizer.cpp` to `MT_SRC`.

- [ ] **Step 4: GREEN**
```
MTD_TEST_GGUF=models/moss-transcribe-f32.gguf MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf ctest --test-dir build -R test_tokenizer --output-on-failure
```
Expected: PASS. If decode≠baseline.text, the vendored tokenizer's special-token handling or byte-map differs — debug against the dumped ids (the byte-BPE decode of the reference tokens must reproduce the text exactly). Do not fudge.

- [ ] **Step 5: Commit** — `git commit -m "M4: Qwen2 byte-BPE tokenizer (decode==baseline.text)"`

---

### Task 11: Audio-span builder + input_ids assembly (M4b) — the decisive M4 gate

**Files:** Create `src/audio_span.{hpp,cpp}`, `tests/test_audio_span.cpp`; modify CMake.

**Interfaces:**
- Consumes: `mt::Tokenizer`, `mt::Config`.
- Produces: `std::vector<int32_t> mt::build_input_ids(const mt::Tokenizer&, const mt::Config&, const std::string& user_prompt, int num_audio_tokens);` — renders the fixed chat-template prompt, expands the single `<|audio_pad|>` into the time-marker-interleaved audio span, returns the full token sequence. Also expose `std::vector<int32_t> mt::audio_span_ids(const mt::Config&, const mt::Tokenizer&, int audio_seq_len);` (the `_audio_span_ids` port) for direct unit testing.

- [ ] **Step 1: Implement `audio_span_ids` (exact port of `processing._audio_span_ids`)**

```cpp
// tokens_per_marker = int(audio_tokens_per_second * time_marker_every_seconds)
// if !enable_time_marker || audio_seq_len<=0 || time_marker_every_seconds<=0: return audio_seq_len copies of audio_token_id
// duration = audio_seq_len / audio_tokens_per_second   (float)
// consumed=0, out=[]
// for sec in range(time_marker_every_seconds, int(duration)+1, time_marker_every_seconds):
//     pos = (sec / time_marker_every_seconds) * tokens_per_marker      // integer division sec/every
//     seg = pos - consumed
//     if seg>0: out += seg copies of audio_token_id; consumed += seg
//     for each decimal digit ch of str(sec): out.push_back(digit_token_ids[ch-'0'])
// remainder = audio_seq_len - consumed; if remainder>0: out += remainder copies of audio_token_id
```
Use `digit_token_ids` from `mt::Config` (indexed 0..9). Match Python integer semantics exactly (`sec // every`, `int(duration)`).

- [ ] **Step 2: Implement `build_input_ids`**
Render the fixed prompt string (Global Constraints), split on the literal `<|audio_pad|>`, encode the `before` and `after` substrings with the tokenizer (add_special_tokens = false semantics — the `<|im_start|>` etc. are matched as special tokens by the tokenizer's special-token pass), and splice `[audio_start_id] + audio_span_ids(num_audio_tokens) + [audio_end_id]` — matching the reference where the chat template emits `<|audio_start|><|audio_pad|><|audio_end|>` and the processor expands only the `<|audio_pad|>`. Confirm the resulting audio_start/end placement matches the reference input_ids.

- [ ] **Step 3: Failing test `tests/test_audio_span.cpp` — bit-exact vs dumped input_ids (BOTH fixtures)**

```cpp
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
```
Register `mt_add_test(test_audio_span)` + `LABELS "model"`.

- [ ] **Step 4: RED then GREEN**
```
MTD_TEST_GGUF=models/moss-transcribe-f32.gguf MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf \
MTD_TEST_BASELINE_LONG=tests/fixtures/baseline_long.gguf ctest --test-dir build -R test_audio_span --output-on-failure
```
Expected: bit-exact for both. This gate simultaneously validates the tokenizer's token IDs (Task 3's ⚠️), the chat-template rendering, and the time-marker interleaver. If it fails, diff `got` vs `ref` around the first mismatch: an early mismatch → tokenizer/prompt text; a mismatch inside the audio span → time-marker math (check `sec//every` and `int(duration)` and `tokens_per_marker`).

- [ ] **Step 5: Commit** — `git commit -m "M4: audio-span builder + input_ids (bit-exact vs reference, both fixtures)"`

---

### Task 12: Qwen3 decoder (M5a)

**Files:** Create `src/qwen3.{hpp,cpp}` (vendored), `src/qwen3_decoder.{hpp,cpp}` (driver), `tests/test_qwen3_decoder.cpp`; modify CMake.

**Interfaces:**
- Produces: `class mt::Qwen3Decoder { bool load(mt::ModelLoader&, int max_seq); void prefill(const std::vector<float>& embeds, int seq, std::vector<float>* all_hidden); std::vector<float> logits_from_hidden(const std::vector<float>& hidden_row); void reset(); int past_len() const; std::vector<float> decode_one(const std::vector<float>& embed); };` — `prefill` returns per-position final-RMSNorm'd hidden states (seq×1024) so the test can check every position; `logits_from_hidden` applies the tied lm_head (`token_embd.weight @ hidden`).

- [ ] **Step 1: Vendor qwen3 layer-forward + loader**
```bash
cp /home/mudler/_git/moss-tts.cpp/src/qwen3.hpp src/qwen3.hpp
cp /home/mudler/_git/moss-tts.cpp/src/qwen3.cpp src/qwen3.cpp
```
Adapt namespace to `mt`. Keep `Qwen3Hparams`, `Qwen3Layer`, `qwen3_load_layer(m, "qwen3", i, &layer)` (reads `qwen3.blk.{i}.*.weight`), and `qwen3_layer_forward(...)` — the NEOX RoPE, per-head q_norm/k_norm before RoPE, GQA-via-broadcast, `GGML_PREC_F32` scores, SwiGLU. Do NOT change the math.

- [ ] **Step 2: Write `src/qwen3_decoder.{hpp,cpp}` (driver)**
Model on moss-tts `DelayBackbone` but read hparams from `mt::Config` (NOT `qwen3.*` KV): build `Qwen3Hparams{hidden=text_hidden, n_heads=text_heads, n_kv_heads=text_kv_heads, head_dim=text_head_dim, intermediate=text_ffn, n_layers=text_layers, text_vocab=text_vocab, rope_base=text_rope_theta, rms_eps=text_rms_eps, use_rope=true}`. Load `token_embd.weight` and `qwen3.output_norm.weight`. Allocate per-layer KV cache tensors `[head_dim, n_kv_heads, max_seq, 1]` on a persistent backend buffer (once). `prefill(embeds[hidden×seq], seq)`: build a no_alloc graph, causal mask `mvec[i*kv+j] = (j<=i)?0:-inf`, run all layers, apply final RMSNorm to ALL positions, capture `[hidden, seq]`. `logits_from_hidden`: `ggml_mul_mat(token_embd_w, hidden)` → `[vocab]` (tied lm_head). `decode_one`: single position at `past_len`, mask=null, update KV cache.

- [ ] **Step 3: Failing test `tests/test_qwen3_decoder.cpp` — feed dumped fused_embeds**

```cpp
#include "qwen3_decoder.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
int main() {
    const char* gguf=std::getenv("MTD_TEST_GGUF"); const char* base=std::getenv("MTD_TEST_BASELINE");
    if(!gguf||!base) return 77;
    mt::ModelLoader m; if(!m.load(gguf)) return 1; m.promote_small_f16_to_f32();
    std::vector<float> fused; std::vector<int64_t> sh;
    if(!mttest::load_baseline(base,"fused_embeds",fused,sh)) return 1;   // [1024, seq]
    const int seq=(int)sh[1], H=(int)sh[0];
    mt::Qwen3Decoder dec; if(!dec.load(m, seq+512)) return 1;
    std::vector<float> hid; dec.prefill(fused, seq, &hid);               // [1024, seq]
    std::vector<float> ref_h; std::vector<int64_t> hsh;
    if(!mttest::load_baseline(base,"lm_hidden",ref_h,hsh)) return 1;
    bool ok = mttest::compare(hid, ref_h, "lm_hidden", 5e-2f, 5e-2f);
    // argmax of last-position logits must match reference greedy first token
    std::vector<float> last(hid.end()-H, hid.end());
    auto logits = dec.logits_from_hidden(last);
    int am=0; for(int i=1;i<(int)logits.size();++i) if(logits[i]>logits[am]) am=i;
    std::vector<float> ref_logits; std::vector<int64_t> lsh;
    if(!mttest::load_baseline(base,"prompt_logits",ref_logits,lsh)) return 1; // [vocab, seq]
    const int V=(int)lsh[0]; int ram=0; const float* rl=&ref_logits[(size_t)(seq-1)*V];
    for(int i=1;i<V;++i) if(rl[i]>rl[ram]) ram=i;
    if(am!=ram){ std::fprintf(stderr,"argmax %d vs ref %d\n",am,ram); ok=false; }
    std::printf("qwen3 lm_hidden + last-pos argmax (%d==%d)\n", am, ram);
    return ok?0:1;
}
```
Register `mt_add_test(test_qwen3_decoder)` + `LABELS "model"`.

- [ ] **Step 4: RED then GREEN**
```
MTD_TEST_GGUF=models/moss-transcribe-f32.gguf MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf ctest --test-dir build -R test_qwen3_decoder --output-on-failure
```
Expected: lm_hidden cos ≥ 0.99999 AND last-position argmax matches. If argmax matches but hidden cosine is borderline, the atol/cos branch still gates it; but the argmax is the decisive check. If it FAILS: most likely RoPE mode (must be NEOX, base 1e6), q/k-norm placement (before RoPE, over head_dim), or GQA broadcast — compare against moss-tts qwen3.cpp op-for-op. Do not loosen.

- [ ] **Step 5: Commit** — `git commit -m "M5: Qwen3 decoder (parity vs lm_hidden + argmax)"`

---

### Task 13: Fusion — embed lookup + masked_scatter (M5b)

**Files:** Create `src/generate.{hpp,cpp}` (fusion part first), `tests/test_fusion.cpp`; modify CMake.

**Interfaces:**
- Produces: `std::vector<float> mt::fuse_embeds(mt::ModelLoader&, const std::vector<int32_t>& input_ids, const std::vector<float>& audio_embeds, int n_audio, int hidden, int audio_token_id);` — embed-lookup `token_embd.weight` rows for `input_ids`, then overwrite the rows at `input_ids[k]==audio_token_id` positions (in order) with the `audio_embeds` rows. Returns `[hidden × seq]` matching the reference `masked_scatter`.

- [ ] **Step 1: Implement fusion**
Embed lookup via `ggml_get_rows(token_embd_w, ids)` OR direct CPU copy of `token_embd.weight` rows (row r = `token_embd[:, r]` since ne=[hidden,vocab]). Then for the j-th audio-token position, copy `audio_embeds[:, j]` over that column. Order: audio positions in increasing index get audio_embeds rows 0,1,2,… (exactly `masked_scatter` semantics). Output `[hidden, seq]`.

- [ ] **Step 2: Failing test `tests/test_fusion.cpp`**
Load `input_ids`, `audio_embeds` from baseline; call `fuse_embeds`; compare to dumped `fused_embeds` at atol 1e-4 / cos ≥ 0.99999 (should be near bit-exact — it's a copy). Register `mt_add_test(test_fusion)` + `LABELS "model"`.

- [ ] **Step 3: RED then GREEN**
```
MTD_TEST_GGUF=... MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf ctest --test-dir build -R test_fusion --output-on-failure
```
Expected: PASS (near bit-exact). A mismatch means either the embed row indexing (token_embd layout) or the audio-position ordering is wrong.

- [ ] **Step 4: Commit** — `git commit -m "M5: masked_scatter fusion (parity vs fused_embeds)"`

---

### Task 14: Audio orchestration + greedy generate + `transcribe` CLI + E2E-vs-upstream (M5c) — GOAL

**Files:** Create `src/audio_encoder.{hpp,cpp}` (chunk/trim/concat orchestration), `src/transcribe.{hpp,cpp}`; extend `src/generate.{hpp,cpp}`; modify `src/cli.cpp`; create `tests/test_generate_e2e.cpp`; modify CMake.

**Interfaces:**
- `class mt::AudioEncoder { AudioEncoder(mt::ModelLoader&); std::vector<float> encode(const std::vector<float>& samples16k, int& n_tokens, int hidden); };` — splits samples into 30s chunks, runs mel+WhisperEncoder+AudioAdaptor per chunk, trims each to `token_len·4` frames pre-merge, concatenates → `[hidden × n_tokens]`.
- `std::vector<int32_t> mt::greedy_generate(mt::Qwen3Decoder&, mt::ModelLoader&, const std::vector<float>& fused, int seq, int max_new, int eos);` — prefill, then loop: argmax last logits → append; embed the new token (token_embd row) → `decode_one` → argmax; stop at eos or max_new.
- `std::string mt::transcribe_wav(mt::ModelLoader&, const std::string& wav_path, int max_new);` — full pipeline: load audio → AudioEncoder → build_input_ids → fuse → greedy_generate → decode new tokens → strip.

- [ ] **Step 1: Implement `AudioEncoder` orchestration**
Reuse `WhisperMel`, `WhisperEncoder`, `AudioAdaptor`. For each 30s chunk: compute `token_len`, pad chunk to 480000, mel→encode→trim to `token_len·4`→ append frames. Concatenate all chunks' trimmed encoder frames → run `AudioAdaptor` once on the full `[1024, ΣtokenLen·4]` → `[1024, Σtoken_len]`. (Matches reference: trim per chunk, concat, then time-merge+adaptor over the whole.)

- [ ] **Step 2: Implement `greedy_generate` and `transcribe_wav`** (generate.cpp / transcribe.cpp) per interfaces.

- [ ] **Step 3: Failing test `tests/test_generate_e2e.cpp` — the decisive gates**

```cpp
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
    mt::Qwen3Decoder dec; dec.load(m, (int)ref_gen.size()+16);
    auto newids = mt::greedy_generate(dec, m, fused, (int)sh[1], (int)(ref_gen.size()-ref_ids.size()), m.config().eos_token_id);
    std::vector<int32_t> ref_new(ref_gen.begin()+(long)ref_ids.size(), ref_gen.end());
    if (newids.size()!=ref_new.size()) { std::fprintf(stderr,"gen len %zu vs %zu\n",newids.size(),ref_new.size()); return 1; }
    for (size_t i=0;i<newids.size();++i) if (newids[i]!=ref_new[i]) { std::fprintf(stderr,"gen mismatch @%zu %d vs %d\n",i,newids[i],ref_new[i]); return 1; }
    // GATE B (E2E vs UPSTREAM): full wav->transcript text == baseline.text
    std::string txt = mt::transcribe_wav(m, "tests/fixtures/short.wav", 300);
    std::string ref; mttest::load_kv_str(base,"baseline.text",ref);
    if (txt != ref) { std::fprintf(stderr,"E2E mismatch:\n got:[%s]\n ref:[%s]\n", txt.c_str(), ref.c_str()); return 1; }
    std::printf("E2E OK: %s\n", txt.c_str());
    return 0;
}
```
Register `mt_add_test(test_generate_e2e)` + `LABELS "model"`.

- [ ] **Step 4: Add `transcribe` CLI subcommand**
`moss-transcribe transcribe <gguf> <wav> [--max-new N]` → prints `transcribe_wav(...)`. Keep it thin.

- [ ] **Step 5: RED then GREEN**
```
MTD_TEST_GGUF=models/moss-transcribe-f32.gguf MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf ctest --test-dir build -R test_generate_e2e --output-on-failure
./build/moss-transcribe transcribe models/moss-transcribe-f32.gguf tests/fixtures/short.wav
```
Expected: GATE A (generated_ids bit-exact) AND GATE B (e2e text == baseline.text) PASS; CLI prints the JFK transcript `[0.28][S01] And so, my fellow Americans,...[10.59]`. **This is the project goal.** If GATE A fails, decode diverges — the argmax at the first divergent step points to which prior component (fusion, decoder, or KV-cache in decode_one) is off. Do not loosen (these are exact-int gates).

- [ ] **Step 6: Run the FULL suite + commit**
```
MTD_TEST_GGUF=models/moss-transcribe-f32.gguf MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf \
MTD_TEST_BASELINE_LONG=tests/fixtures/baseline_long.gguf ctest --test-dir build -L model --output-on-failure
```
Expected: all model tests pass (mel, encoder, adaptor, front-end e2e, tokenizer, audio_span, qwen3_decoder, fusion, generate_e2e). Commit — `git commit -m "M5: audio orchestration + greedy generate + transcribe CLI (E2E == upstream)"`.

---

## Self-Review

**Spec coverage (M4–M5):** tokenizer → Task 10; audio-span + time markers (bit-exact input_ids) → Task 11; Qwen3 decoder parity → Task 12; masked_scatter fusion → Task 13; orchestration + greedy generate + CLI + e2e-vs-upstream → Task 14. The e2e-vs-upstream gate (GATE B, text==baseline.text) is the explicit acceptance criterion. Multi-chunk (long) validated at the input_ids level (Task 11) and orchestration level (Task 14 AudioEncoder handles chunks); a long-fixture generate gate is optional (CPU-slow) and not required for the goal.

**Placeholder scan:** Vendored files (tokenizer, qwen3) are copied with named adaptations; the driver (`qwen3_decoder`), fusion, orchestration, and generate loop have concrete interfaces and gated tests. The `audio_span_ids` port gives the exact integer algorithm. No "TBD"/"handle edge cases".

**Type consistency:** `mt::Tokenizer`, `mt::build_input_ids`, `mt::Qwen3Decoder` (`load`/`prefill`/`logits_from_hidden`/`decode_one`), `mt::fuse_embeds`, `mt::AudioEncoder`, `mt::greedy_generate`, `mt::transcribe_wav` signatures are consistent across defining and consuming tasks. GGUF tensor/KV names match Phase 1's converter.

## Reaching the goal
Task 14 GATE B (CLI transcribes `short.wav` to text equal to the genuine upstream `baseline.text`) is the working end-to-end transcription from a fixture. M6–M10 (subtitle/diarization export, C-API, GPU, quant, LocalAI) are beyond the goal and get their own plans on request.
