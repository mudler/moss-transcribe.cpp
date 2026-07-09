# moss-transcribe.cpp — Phase 1: Foundation & Audio Front-End (M0–M3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the repo, the metadata-driven GGUF converter, the reference-dump parity harness, and the audio front-end path (`audio_io → mel → whisper_encoder → audio_adaptor`) proven numerically equal to the PyTorch reference — i.e. produce correct audio embeds `(N,1024)` from a wav.

**Architecture:** Standalone C++17/ggml repo mirroring `parakeet.cpp`'s package shape. ggml is a submodule; the Qwen3 decoder, Qwen2 BPE tokenizer, and ggml/backend/loader helpers are vendored (copy-and-adapt) from `moss-tts.cpp`. Every dimension/hyperparameter lives in one self-contained GGUF; the loader reads them, nothing is hardcoded. Each component is gated F32-on-CPU against reference tensors dumped from the real model before the next component starts.

**Tech Stack:** C++17, CMake ≥3.18, ggml (submodule), Python 3.12 + `gguf`/`transformers`/`torch`/`safetensors`/`librosa` for the converter and baseline generator.

## Global Constraints

- Project namespace: `mt::`. GGUF metadata KV prefix: `mtd.`. C-API symbol prefix: `moss_transcribe_capi_`. Library/target: `moss-transcribe`. CLI binary: `moss-transcribe`. Test env vars: `MTD_TEST_GGUF`, `MTD_TEST_BASELINE`. Device override env: `MTD_DEVICE`.
- Parity build/gate is **F32 on CPU**. No GPU/quant work in Phase 1.
- `general.architecture = "moss_transcribe_diarize"`.
- Exact reference constants (from HF `config.json`/`preprocessor_config.json`, cross-checked vs vLLM PR #47729) — copy verbatim:
  - Text (Qwen3): vocab 151936, hidden 1024, ffn 3072, 28 layers, 16 Q heads, 8 KV heads, **head_dim 128** (independent of hidden/n_heads), rms_eps 1e-6, rope_theta 1e6, tied lm_head.
  - Audio (Whisper): mel_bins 80, d_model 1024, 24 layers, 16 heads (**head_dim 64**), ffn 4096, max_source_positions 1500, activation **gelu (erf, exact — not tanh)**, scale_embedding false. Attention: q_proj+bias, **k_proj no bias**, v_proj+bias, out_proj+bias.
  - Bridge: audio_token_id 151671, audio_merge_size 4, adaptor_input_dim 4096, adaptor LayerNorm eps 1e-6.
  - Mel: sr 16000, n_fft 400, hop 160, feature_size 80, n_samples 480000, nb_max_frames 3000, dither 0, center STFT (reflect pad), periodic Hann, power spectrum, slaney mel filters (80×201), log10, per-chunk normalization `x=max(x, x.max()-8); x=(x+4)/4`.
  - Audio-span: audio_tokens_per_second 12.5, time_marker_every_seconds 2, enable_time_marker true.
  - Generation: eos 151645, pad 151643, default_max_new_tokens 5120.
- Float parity tolerance: cosine ≥ 0.99999 AND max-abs-err ≤ small multiple of tensor std. Integer tensors bit-exact. Isolated-stage `atol` 1e-3; full-encoder `atol` 5e-2 (deep-transformer accumulation).
- Reference dumps and the converter's mel filterbank are stored **inside GGUF** (baseline as `baseline.gguf`, keyed by tensor name), read back via `tests/parity.hpp` — same convention as parakeet.cpp.
- DRY, YAGNI, TDD, frequent commits. Each task ends green and is committed.

## File Structure (Phase 1)

Created/touched in this phase (later phases add `qwen3.cpp`, `tokenizer.cpp`, `audio_span.cpp`, `generate.cpp`, `transcript_parser.cpp`, `subtitle.cpp`, `cli.cpp` transcribe path, C-API, LocalAI backend):

```
CMakeLists.txt                     # lib + cli + tests wiring, ggml submodule
.gitmodules                        # third_party/ggml
third_party/ggml/                  # submodule
include/moss_transcribe.h          # public C++ API (grows over phases; Phase 1: Config + loader entry)
src/common.{hpp,cpp}               # logging, file utils (vendored+trimmed from moss-tts)
src/ggml_extend.hpp                # ctx RAII, linear/layer_norm helpers (vendored from moss-tts)
src/backend.{hpp,cpp}              # ggml backend singleton + compute (vendored+adapted from moss-tts)
src/model_loader.{hpp,cpp}         # GGUF reader + Config struct (adapted; mtd.* keys)
src/config.hpp                     # mt::Config (all dims read from GGUF)
src/audio_io.{hpp,cpp}             # wav -> 16k mono f32
src/fft.{hpp,cpp}                  # CPU real FFT (radix-2)
src/mel.{hpp,cpp}                  # Whisper log-mel (loads mel_filters from GGUF)
src/whisper_encoder.{hpp,cpp}      # conv stem + 24 layers + final LN
src/audio_adaptor.{hpp,cpp}        # 4x time-merge + VQAdaptor
src/cli.cpp                        # Phase 1: `info` subcommand only
scripts/convert_moss_transcribe_to_gguf.py
scripts/gen_baseline.py
scripts/requirements.txt
tests/CMakeLists.txt
tests/parity.hpp                   # baseline.gguf loader + compare() (vendored from parakeet)
tests/test_mel.cpp
tests/test_whisper_encoder.cpp
tests/test_audio_adaptor.cpp
tests/python/check_convert.py
```

---

### Task 1: Repo scaffold + build system (M0a)

Bootstrap a buildable empty library + `info`-less CLI stub so every later task has a green baseline to build against.

**Files:**
- Create: `.gitmodules`, `CMakeLists.txt`, `src/common.hpp`, `src/common.cpp`, `src/ggml_extend.hpp`, `src/cli.cpp`, `include/moss_transcribe.h`, `.gitignore`
- Submodule: `third_party/ggml`

**Interfaces:**
- Produces: library target `moss-transcribe`; CLI binary `moss-transcribe`; `MT_BUILD_TESTS` CMake option (default OFF); `mt::` namespace; logging macros `MT_LOGE/W/I/D`.

- [ ] **Step 1: Add ggml submodule (pinned) and .gitmodules**

```bash
cd /home/mudler/_git/moss-transcribe.cpp
git submodule add https://github.com/ggml-org/ggml third_party/ggml
cd third_party/ggml && git checkout master && git rev-parse HEAD && cd ../..
# Pin: record the SHA in .gitmodules comment and commit the gitlink
```
`.gitmodules` must read:
```
[submodule "third_party/ggml"]
	path = third_party/ggml
	url = https://github.com/ggml-org/ggml
```

- [ ] **Step 2: Write `.gitignore`**

```
/build/
/models/
*.gguf
__pycache__/
.venv/
```

- [ ] **Step 3: Write `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.18)
project(moss_transcribe CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(MT_BUILD_TESTS "Build tests" OFF)
option(MT_BUILD_CLI "Build CLI" ON)
option(MT_SHARED "Build shared lib" OFF)
foreach(be CUDA METAL VULKAN HIP)
  option(MT_GGML_${be} "Enable ggml ${be}" OFF)
  if(MT_GGML_${be})
    set(GGML_${be} ON CACHE BOOL "" FORCE)
  endif()
endforeach()
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml)

set(MT_SRC
  src/common.cpp
)
if(MT_SHARED)
  add_library(moss-transcribe SHARED ${MT_SRC})
  set_target_properties(moss-transcribe PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(moss-transcribe STATIC ${MT_SRC})
endif()
target_include_directories(moss-transcribe PUBLIC include PRIVATE src third_party)
target_link_libraries(moss-transcribe PUBLIC ggml)

if(MT_BUILD_CLI)
  add_executable(moss-transcribe-cli src/cli.cpp)
  set_target_properties(moss-transcribe-cli PROPERTIES OUTPUT_NAME moss-transcribe)
  target_link_libraries(moss-transcribe-cli PRIVATE moss-transcribe)
endif()

if(MT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
```
(`MT_SRC` gains one file per later task.)

- [ ] **Step 4: Vendor `src/common.{hpp,cpp}` and `src/ggml_extend.hpp`**

Copy from moss-tts.cpp and trim the `include/moss_tts.h` dependency:
```bash
cp /home/mudler/_git/moss-tts.cpp/src/common.hpp     src/common.hpp
cp /home/mudler/_git/moss-tts.cpp/src/common.cpp     src/common.cpp
cp /home/mudler/_git/moss-tts.cpp/src/ggml_extend.hpp src/ggml_extend.hpp
```
Adapt: replace the `moss_log_level` enum dependency with a local enum in `common.hpp`, rename macros `MOSS_LOG*`→`MT_LOG*`, put everything in `namespace mt`. `ggml_extend.hpp` provides `mt::make_ctx`, `mt::make_ctx_buf`, `mt::linear(ctx,W,b,x)`, `mt::layer_norm(ctx,x,w,b,eps)` — keep these signatures.

- [ ] **Step 5: Write `include/moss_transcribe.h` (Phase 1 minimal) and `src/cli.cpp` stub**

`include/moss_transcribe.h`:
```cpp
#pragma once
namespace mt { const char* version(); }
```
`src/common.cpp` add: `const char* mt::version() { return "0.0.1"; }`
`src/cli.cpp`:
```cpp
#include "moss_transcribe.h"
#include <cstdio>
#include <cstring>
int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: moss-transcribe <subcommand>\n"); return 2; }
    if (std::strcmp(argv[1], "version") == 0) { std::printf("%s\n", mt::version()); return 0; }
    std::fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    return 2;
}
```

- [ ] **Step 6: Configure, build, run**

Run:
```bash
cmake -B build -DMT_BUILD_TESTS=OFF && cmake --build build -j
./build/moss-transcribe version
```
Expected: prints `0.0.1`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "M0: repo scaffold, ggml submodule, vendored common/ggml_extend, CLI stub"
```

---

### Task 2: Backend + ModelLoader + Config (M0b)

Bring in the GGUF loader and backend so we can read a model file and its metadata.

**Files:**
- Create: `src/backend.hpp`, `src/backend.cpp`, `src/model_loader.hpp`, `src/model_loader.cpp`, `src/config.hpp`
- Modify: `CMakeLists.txt` (`MT_SRC` += backend.cpp, model_loader.cpp)

**Interfaces:**
- Produces:
  - `mt::backend()` → `ggml_backend_t` (lazy singleton; env `MTD_DEVICE`), `mt::compute_graph_with_inputs(ggml_cgraph*, const std::function<void()>&)`, `mt::allocate_ctx_tensors(ggml_context*)`, `mt::read_tensor_f32(const ggml_tensor*, std::vector<float>*)`, `mt::read_tensor_i32(...)`.
  - `struct mt::Config` (fields below), read by `mt::ModelLoader`.
  - `class mt::ModelLoader { bool load(const std::string& path); const Config& config() const; ggml_tensor* tensor(const std::string& name) const; bool has(const std::string&) const; void promote_small_f16_to_f32(size_t max_elems=65536); ggml_context* ggml_ctx() const; /* kv getters */ int64_t get_i64(const char*,int64_t=0); uint32_t get_u32(const char*,uint32_t=0); float get_f32(const char*,float=0); bool get_bool(const char*,bool=false); std::string get_str(const char*,std::string=""); std::vector<int32_t> get_i32_array(const char*); std::vector<std::string> get_str_array(const char*); };`

- [ ] **Step 1: Vendor backend + model_loader from moss-tts, adapt prefixes**

```bash
cp /home/mudler/_git/moss-tts.cpp/src/backend.hpp      src/backend.hpp
cp /home/mudler/_git/moss-tts.cpp/src/backend.cpp      src/backend.cpp
cp /home/mudler/_git/moss-tts.cpp/src/model_loader.hpp src/model_loader.hpp
cp /home/mudler/_git/moss-tts.cpp/src/model_loader.cpp src/model_loader.cpp
```
Adapt: namespace → `mt`; env `MOSS_TTS_BACKEND`→`MTD_DEVICE`; compile flags `MOSS_TTS_HAVE_*`→`MT_HAVE_*`; keep the public signatures listed in Interfaces. Keep `promote_small_f16_to_f32` (norms/biases must be f32 on CPU).

- [ ] **Step 2: Write `src/config.hpp` — Config read from `mtd.*` KV**

```cpp
#pragma once
#include <string>
#include <vector>
namespace mt {
struct Config {
    std::string arch;
    // text (qwen3)
    int text_vocab, text_hidden, text_ffn, text_layers, text_heads, text_kv_heads, text_head_dim;
    float text_rms_eps, text_rope_theta; int text_max_pos; bool text_tied;
    // audio (whisper)
    int audio_mel_bins, audio_d_model, audio_layers, audio_heads, audio_ffn, audio_max_src_pos;
    std::string audio_act; bool audio_scale_embed;
    // bridge
    int audio_token_id, audio_merge_size, adaptor_input_dim; float adaptor_norm_eps;
    // mel
    int feat_sr, feat_n_fft, feat_hop, feat_size, feat_n_samples, feat_nb_max_frames; float feat_dither;
    // audio-span
    float audio_tokens_per_second; int time_marker_every_seconds; bool enable_time_marker;
    // generation
    int eos_token_id, pad_token_id, default_max_new_tokens; std::string default_prompt;
    std::vector<int32_t> digit_token_ids; // 10 ids for "0".."9"
};
} // namespace mt
```

- [ ] **Step 3: In `model_loader.cpp`, populate `Config` from KV inside `load()`**

Add a `read_config()` reading each key (mirror parakeet's `kv_*` helpers). Example lines (repeat for every field):
```cpp
cfg_.text_hidden   = (int) get_u32("mtd.text.hidden", 1024);
cfg_.text_head_dim = (int) get_u32("mtd.text.head_dim", 128);
cfg_.audio_d_model = (int) get_u32("mtd.audio.d_model", 1024);
cfg_.audio_act     = get_str("mtd.audio.act", "gelu");
cfg_.audio_token_id= (int) get_u32("mtd.audio_token_id", 151671);
cfg_.audio_merge_size = (int) get_u32("mtd.audio_merge_size", 4);
cfg_.adaptor_input_dim= (int) get_u32("mtd.adaptor_input_dim", 4096);
cfg_.adaptor_norm_eps = get_f32("mtd.adaptor_norm_eps", 1e-6f);
cfg_.feat_n_fft    = (int) get_u32("mtd.feat.n_fft", 400);
cfg_.feat_hop      = (int) get_u32("mtd.feat.hop", 160);
cfg_.feat_size     = (int) get_u32("mtd.feat.feature_size", 80);
cfg_.feat_n_samples= (int) get_u32("mtd.feat.n_samples", 480000);
cfg_.feat_nb_max_frames = (int) get_u32("mtd.feat.nb_max_frames", 3000);
cfg_.audio_tokens_per_second = get_f32("mtd.audio_tokens_per_second", 12.5f);
cfg_.time_marker_every_seconds = (int) get_u32("mtd.time_marker_every_seconds", 2);
cfg_.digit_token_ids = get_i32_array("mtd.digit_token_ids");
// ... all remaining fields, defaults = the Global Constraints values
cfg_.arch = get_str("general.architecture", "");
```

- [ ] **Step 4: Add to `CMakeLists.txt`**

`MT_SRC` becomes:
```cmake
set(MT_SRC src/common.cpp src/backend.cpp src/model_loader.cpp)
```
Build to confirm it still compiles:
```bash
cmake --build build -j
```
Expected: builds clean (no test yet — loader is exercised in Task 4's `info`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "M0: backend + GGUF ModelLoader + Config (mtd.* metadata)"
```

---

### Task 3: GGUF converter (M0c)

Convert the HF safetensors + config/tokenizer into one self-contained `mtd.*` GGUF. Gate is a Python check that the produced file has the required KV + tensors and correct shapes.

**Files:**
- Create: `scripts/convert_moss_transcribe_to_gguf.py`, `scripts/requirements.txt`, `tests/python/check_convert.py`
- Modify: `tests/CMakeLists.txt` (created here)

**Interfaces:**
- Produces: a GGUF with all `mtd.*` KV (Global Constraints), tokenizer KV (`tokenizer.tokens/token_type/merges/special_tokens_ids/special_tokens_text/bos_id/eos_id/pad_id`), `mtd.digit_token_ids`, `mel_filters` (80×201 f32), and tensors named per the spec's §3 (encoder `enc.*`, `adaptor.*`, decoder `blk.*`/`token_embd.w`). `--dtype {f32,f16,q8_0}` (default f32).

- [ ] **Step 1: Write `scripts/requirements.txt`**

```
torch>=2.8
transformers>=5.0.0,<6.0.0
safetensors>=0.6.2
gguf
numpy>=1.26,<3
librosa>=0.11.0
```

- [ ] **Step 2: Write `scripts/convert_moss_transcribe_to_gguf.py`**

Structure (concrete, ~200 lines — key parts shown, fill mechanically for every tensor group):
```python
import argparse, json, numpy as np, torch, gguf
from transformers import AutoConfig, AutoProcessor
from safetensors.torch import load_file

TEXT, AUDIO = "text", "audio"

def add_metadata(w, cfg):
    w.add_string("general.architecture", "moss_transcribe_diarize")
    t = cfg.text_config; a = cfg.audio_config
    w.add_uint32("mtd.text.vocab_size", t.vocab_size)
    w.add_uint32("mtd.text.hidden", t.hidden_size)
    w.add_uint32("mtd.text.ffn", t.intermediate_size)
    w.add_uint32("mtd.text.n_layers", t.num_hidden_layers)
    w.add_uint32("mtd.text.n_heads", t.num_attention_heads)
    w.add_uint32("mtd.text.n_kv_heads", t.num_key_value_heads)
    w.add_uint32("mtd.text.head_dim", t.head_dim)
    w.add_float32("mtd.text.rms_eps", t.rms_norm_eps)
    w.add_float32("mtd.text.rope_theta", float(t.rope_theta))
    w.add_uint32("mtd.text.max_pos", t.max_position_embeddings)
    w.add_bool("mtd.text.tied", True)
    w.add_uint32("mtd.audio.mel_bins", a.num_mel_bins)
    w.add_uint32("mtd.audio.d_model", a.d_model)
    w.add_uint32("mtd.audio.n_layers", a.encoder_layers)
    w.add_uint32("mtd.audio.n_heads", a.encoder_attention_heads)
    w.add_uint32("mtd.audio.ffn", a.encoder_ffn_dim)
    w.add_uint32("mtd.audio.max_src_pos", a.max_source_positions)
    w.add_string("mtd.audio.act", a.activation_function)
    w.add_bool("mtd.audio.scale_embed", bool(a.scale_embedding))
    w.add_uint32("mtd.audio_token_id", cfg.audio_token_id)
    w.add_uint32("mtd.audio_merge_size", cfg.audio_merge_size)
    w.add_uint32("mtd.adaptor_input_dim", cfg.adaptor_input_dim)
    w.add_float32("mtd.adaptor_norm_eps", t.rms_norm_eps)

def add_feature_extractor(w, fe, proc):
    w.add_uint32("mtd.feat.sr", fe.sampling_rate)
    w.add_uint32("mtd.feat.n_fft", fe.n_fft)
    w.add_uint32("mtd.feat.hop", fe.hop_length)
    w.add_uint32("mtd.feat.feature_size", fe.feature_size)
    w.add_uint32("mtd.feat.n_samples", fe.n_samples)
    w.add_uint32("mtd.feat.nb_max_frames", fe.nb_max_frames)
    w.add_float32("mtd.feat.dither", float(getattr(fe, "dither", 0.0)))
    w.add_float32("mtd.audio_tokens_per_second", float(proc.audio_tokens_per_second))
    w.add_uint32("mtd.time_marker_every_seconds", int(proc.time_marker_every_seconds))
    w.add_bool("mtd.enable_time_marker", bool(proc.enable_time_marker))
    # slaney mel filters (80 x 201) exactly as HF computes them:
    mel = np.asarray(fe.mel_filters, dtype=np.float32)   # shape (201, 80) in HF
    w.add_tensor("mel_filters", np.ascontiguousarray(mel.T))  # store (80, 201) row-major

def add_tokenizer(w, tok, proc):
    vocab = tok.get_vocab()  # token->id
    id2tok = {i: t for t, i in vocab.items()}
    n = max(id2tok) + 1
    tokens = [id2tok.get(i, "") for i in range(n)]
    w.add_array("tokenizer.tokens", tokens)
    # token_type, merges, special ids/texts, bos/eos/pad — mirror moss-tts convert_tokenizer.py
    # digit ids (parity with processor._get_digit_token_ids):
    digits = [tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"]
    w.add_array("mtd.digit_token_ids", [int(x) for x in digits])
    w.add_uint32("mtd.eos_token_id", 151645)
    w.add_uint32("mtd.pad_token_id", 151643)
    w.add_uint32("mtd.default_max_new_tokens", 5120)
    w.add_string("mtd.default_prompt", DEFAULT_PROMPT)  # copy from inference_utils.DEFAULT_PROMPT

RENAME = [
    # (regex/prefix, replacement) mapping HF -> ggml names; see spec §3
    ("model.whisper_encoder.conv1", "enc.conv1"),
    ("model.whisper_encoder.conv2", "enc.conv2"),
    ("model.whisper_encoder.embed_positions.weight", "enc.pos_embd"),
    ("model.whisper_encoder.layer_norm", "enc.ln_post"),
    # enc.blk.{i}.* , adaptor.* , blk.{i}.* , token_embd.w  (fill all, see helper below)
]

def rename(name):  # returns ggml tensor name; implement the full mapping per spec §3
    ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--dtype", choices=["f32","f16","q8_0"], default="f32")
    args = ap.parse_args()
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    sd = {}
    # load all safetensors shards
    for shard in sorted(glob(f"{args.model_dir}/model*.safetensors")):
        sd.update(load_file(shard))
    w = gguf.GGUFWriter(args.out, "moss_transcribe_diarize")
    add_metadata(w, cfg); add_feature_extractor(w, proc.feature_extractor, proc)
    add_tokenizer(w, proc.tokenizer, proc)
    for hf_name, tensor in sd.items():
        if hf_name == "lm_head.weight":   # tied -> skip, decoder aliases token_embd
            continue
        arr = tensor.to(torch.float32).numpy()
        gname = rename(hf_name)
        # dtype handling: f32 default; f16/q8_0 only for 2D matmul weights >= 32 in both dims
        w.add_tensor(gname, np.ascontiguousarray(arr))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
```
Notes to implement precisely: (a) `fe.mel_filters` in HF is shape `(1+n_fft//2, n_mels)` = (201,80); store transposed to (80,201). (b) conv weights: HF `Conv1d.weight` is `(out,in,k)` — keep as-is; the C++ side adapts for `ggml_conv_1d_ph`. (c) implement `rename()` for **every** tensor listed in spec §3; assert at the end that all expected ggml names are present.

- [ ] **Step 3: Download weights and run the converter (real model)**

Run:
```bash
python -m pip install -r scripts/requirements.txt
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize --local-dir models/hf
python scripts/convert_moss_transcribe_to_gguf.py models/hf -o models/moss-transcribe-f32.gguf --dtype f32
```
Expected: writes `models/moss-transcribe-f32.gguf` (~3.6 GB f32).

- [ ] **Step 4: Write `tests/python/check_convert.py` (the gate)**

```python
import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
kv = {f.name: f for f in r.fields.values()}
need_kv = ["general.architecture","mtd.text.hidden","mtd.audio.d_model",
           "mtd.audio_token_id","mtd.feat.n_fft","mtd.digit_token_ids",
           "mtd.audio_tokens_per_second","tokenizer.tokens"]
for k in need_kv:
    assert k in kv, f"missing KV {k}"
names = {t.name for t in r.tensors}
need_t = ["mel_filters","enc.conv1.w","enc.conv2.w","enc.pos_embd","enc.ln_post.w",
          "enc.blk.0.attn_q.w","enc.blk.23.ffn_2.w",
          "adaptor.fc1.w","adaptor.fc2.w","adaptor.ln.w",
          "token_embd.w","blk.0.attn_q.w","blk.27.ffn_down.w","output_norm.w"]
for t in need_t:
    assert t in names, f"missing tensor {t}"
mf = next(t for t in r.tensors if t.name == "mel_filters")
assert list(mf.shape) == [201, 80] or list(mf.shape) == [80, 201], mf.shape
print("OK", len(names), "tensors")
```

- [ ] **Step 5: Run the gate**

Run:
```bash
python tests/python/check_convert.py models/moss-transcribe-f32.gguf
```
Expected: prints `OK <n> tensors`, exit 0.

- [ ] **Step 6: Wire `info` subcommand + build `tests/CMakeLists.txt` skeleton**

In `src/cli.cpp` add an `info` subcommand:
```cpp
// argv[1]=="info", argv[2]=gguf path
#include "model_loader.hpp"
...
mt::ModelLoader m;
if (!m.load(argv[2])) { std::fprintf(stderr, "load failed\n"); return 1; }
const auto& c = m.config();
std::printf("arch=%s text.hidden=%d text.layers=%d audio.d_model=%d audio.layers=%d\n",
            c.arch.c_str(), c.text_hidden, c.text_layers, c.audio_d_model, c.audio_layers);
std::printf("audio_token_id=%d merge=%d adaptor_in=%d mel_bins=%d n_fft=%d hop=%d\n",
            c.audio_token_id, c.audio_merge_size, c.adaptor_input_dim, c.feat_size, c.feat_n_fft, c.feat_hop);
```
Link `moss-transcribe-cli` also against loader (already via `moss-transcribe`). Create `tests/CMakeLists.txt` with the parakeet-style helper (used from Task 4 on):
```cmake
macro(mt_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE moss-transcribe)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endmacro()
```

- [ ] **Step 7: Run `info`, commit**

Run:
```bash
cmake --build build -j
./build/moss-transcribe info models/moss-transcribe-f32.gguf
```
Expected: prints `arch=moss_transcribe_diarize text.hidden=1024 text.layers=28 audio.d_model=1024 audio.layers=24 ...`.
```bash
git add -A
git commit -m "M0: GGUF converter + check_convert gate + info subcommand"
```

---

### Task 4: Parity harness — gen_baseline.py + parity.hpp (M1a)

Dump reference tensors from the real model into `baseline.gguf`, and add the C++ reader/compare helper. This unblocks every component gate.

**Files:**
- Create: `scripts/gen_baseline.py`, `tests/parity.hpp`
- Modify: `tests/CMakeLists.txt` (nothing new yet; helper already present)

**Interfaces:**
- Produces: `baseline.gguf` with named f32/i32 tensors: `input_features` (n_chunk,80,3000), `audio_feature_lengths` (i32), `audio_chunk_mapping` (i32), `input_ids` (i32), `encoder_hidden` (n_chunk,1500,1024), `merged` (N,4096), `audio_embeds` (N,1024), `fused_embeds` (seq,1024), `lm_hidden` (seq,1024), `prompt_logits` (seq,vocab), `generated_ids` (i32), and KV `baseline.text` (str). Two baselines: `baseline_short.gguf`, `baseline_long.gguf`.
- `tests/parity.hpp` (namespace `mttest`): `bool load_baseline(path,name,std::vector<float>&,std::vector<int64_t>& shape)`, `bool load_baseline_i32(path,name,std::vector<int32_t>&)`, `bool compare(const std::vector<float>& got,const std::vector<float>& ref,const char* label,float atol,float rtol)`, `bool load_kv_str(path,key,std::string&)`.

- [ ] **Step 1: Vendor `tests/parity.hpp` from parakeet, adapt namespace**

```bash
cp /home/mudler/_git/parakeet.cpp/tests/parity.hpp tests/parity.hpp
```
Adapt: `namespace pktest`→`mttest`. `compare()` computes and prints max/mean abs-diff AND cosine similarity; returns pass if `max_abs<=atol OR (cos>=0.99999 AND max_abs<=rtol*std)`. Add cosine to `compare` if not present:
```cpp
// inside compare(): double dot=0,na=0,nb=0; for(i) {dot+=got[i]*ref[i]; na+=got[i]*got[i]; nb+=ref[i]*ref[i];}
// double cos = dot / (sqrt(na)*sqrt(nb)+1e-30);
```

- [ ] **Step 2: Write `scripts/gen_baseline.py`**

```python
import argparse, numpy as np, torch, gguf
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("wav"); ap.add_argument("-o", required=True)
    args = ap.parse_args()
    device = torch.device("cpu"); dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True,
             dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    msgs = build_transcription_messages(args.wav)
    inputs = prepare_inputs(proc, msgs, device=device)
    w = gguf.GGUFWriter(args.o, "moss_transcribe_diarize_baseline")

    def dump(name, t):
        a = t.detach().cpu().to(torch.float32).numpy() if t.dtype.is_floating_point \
            else t.detach().cpu().to(torch.int32).numpy()
        w.add_tensor(name, np.ascontiguousarray(a))

    with torch.no_grad():
        dump("input_features", inputs["input_features"])
        dump("audio_feature_lengths", inputs["audio_feature_lengths"])
        dump("audio_chunk_mapping", inputs["audio_chunk_mapping"])
        dump("input_ids", inputs["input_ids"][0])
        mdl = model.model
        enc = mdl.whisper_encoder(inputs["input_features"].to(dtype), return_dict=True).last_hidden_state
        dump("encoder_hidden", enc)
        feats = model.get_audio_features(inputs["input_features"], inputs["audio_feature_lengths"],
                                         inputs.get("audio_chunk_mapping"))
        audio_embeds = torch.cat([f.squeeze(0) for f in feats], dim=0)
        dump("audio_embeds", audio_embeds)
        # merged (pre-adaptor): replicate trim+concat+time_merge for a single audio
        # (call mdl.time_merge on the concatenated trimmed encoder output)
        # ... dump("merged", merged)
        embeds = mdl.get_input_embeddings()(inputs["input_ids"])
        fused = mdl.inject_audio_features(inputs["input_ids"], embeds, inputs["input_features"],
                    inputs["audio_feature_lengths"], inputs.get("audio_chunk_mapping"))
        dump("fused_embeds", fused[0])
        out = model(inputs_embeds=fused, attention_mask=inputs["attention_mask"], return_dict=True)
        dump("prompt_logits", out.logits[0])
        gen = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"], audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs.get("audio_chunk_mapping"), do_sample=False, max_new_tokens=256)
        dump("generated_ids", gen[0])
        text = proc.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        w.add_string("baseline.text", text)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", args.o)
```
Also compute `lm_hidden` by capturing `out.hidden_states[-1]` (pass `output_hidden_states=True`) and `dump("lm_hidden", ...)`.

- [ ] **Step 3: Prepare fixtures and generate both baselines**

Run:
```bash
mkdir -p tests/fixtures
cp /home/mudler/_git/whisper.cpp/samples/jfk.wav tests/fixtures/short.wav
# long fixture: concatenate short 4x (>30s) to force multi-chunk + time markers
python - <<'PY'
import soundfile as sf, numpy as np
x, sr = sf.read("tests/fixtures/short.wav")
sf.write("tests/fixtures/long.wav", np.tile(x, 4), sr)
PY
pip install -e scripts/../  # install the reference package (moss_transcribe_diarize) for inference_utils
python scripts/gen_baseline.py models/hf tests/fixtures/short.wav -o tests/fixtures/baseline_short.gguf
python scripts/gen_baseline.py models/hf tests/fixtures/long.wav  -o tests/fixtures/baseline_long.gguf
```
Expected: both `baseline_*.gguf` written; prints `wrote ...`.

- [ ] **Step 4: Sanity check baseline contents**

Run:
```bash
python - <<'PY'
import gguf
r = gguf.GGUFReader("tests/fixtures/baseline_short.gguf")
print({t.name: list(t.shape) for t in r.tensors})
PY
```
Expected: `input_features [1,80,3000]`, `encoder_hidden [1,1500,1024]`, `audio_embeds [~,1024]`, `input_ids [~]`, `generated_ids [~]` present.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_baseline.py tests/parity.hpp tests/CMakeLists.txt
git commit -m "M1: parity harness — gen_baseline.py + parity.hpp; baseline fixtures"
```

---

### Task 5: audio_io — wav → 16 kHz mono f32 (M1b)

**Files:**
- Create: `src/audio_io.hpp`, `src/audio_io.cpp`
- Modify: `CMakeLists.txt` (`MT_SRC += src/audio_io.cpp`)

**Interfaces:**
- Produces: `struct mt::Audio { std::vector<float> samples; int sample_rate; };`, `bool mt::load_audio_16k_mono(const std::string& path, Audio& out);`, `std::vector<float> mt::resample_linear(const std::vector<float>&, int in_sr, int out_sr);`

- [ ] **Step 1: Vendor audio_io from parakeet**

```bash
cp /home/mudler/_git/parakeet.cpp/src/audio_io.hpp src/audio_io.hpp
cp /home/mudler/_git/parakeet.cpp/src/audio_io.cpp src/audio_io.cpp
```
Adapt namespace to `mt`. (WAV parse + linear resample to 16k mono; no code change needed beyond namespace.)

- [ ] **Step 2: Write failing test `tests/test_audio_io.cpp`**

```cpp
#include "audio_io.hpp"
#include <cstdio>
int main() {
    const char* wav = "tests/fixtures/short.wav";
    mt::Audio a;
    if (!mt::load_audio_16k_mono(wav, a)) { std::fprintf(stderr, "load failed\n"); return 1; }
    if (a.sample_rate != 16000) { std::fprintf(stderr, "sr=%d\n", a.sample_rate); return 1; }
    if (a.samples.size() < 16000) { std::fprintf(stderr, "too short: %zu\n", a.samples.size()); return 1; }
    std::printf("ok samples=%zu sr=%d\n", a.samples.size(), a.sample_rate);
    return 0;
}
```
Register in `tests/CMakeLists.txt`: `mt_add_test(test_audio_io)`.

- [ ] **Step 3: Run to verify it fails (not yet in MT_SRC)**

Run:
```bash
cmake -B build -DMT_BUILD_TESTS=ON && cmake --build build -j 2>&1 | tail -5
```
Expected: link error (undefined `mt::load_audio_16k_mono`) — audio_io.cpp not in MT_SRC yet.

- [ ] **Step 4: Add to MT_SRC, build, run**

Add `src/audio_io.cpp` to `MT_SRC`. Run:
```bash
cmake --build build -j && ctest --test-dir build -R test_audio_io --output-on-failure
```
Expected: PASS (`ok samples=... sr=16000`).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M1: audio_io (wav->16k mono)"
```

---

### Task 6: mel — Whisper log-mel front-end (M1c)

**Files:**
- Create: `src/fft.hpp`, `src/fft.cpp`, `src/mel.hpp`, `src/mel.cpp`, `tests/test_mel.cpp`
- Modify: `CMakeLists.txt` (`MT_SRC += fft.cpp mel.cpp`), `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `mt::ModelLoader` (reads `mel_filters` tensor + `feat.*` config), `mt::Audio`.
- Produces: `void mt::rfft(const std::vector<float>& in, std::vector<float>& re, std::vector<float>& im);` and `class mt::WhisperMel { explicit WhisperMel(const mt::ModelLoader&); void compute(const std::vector<float>& samples, std::vector<float>& out, int& n_mels, int& n_frames) const; };` — `out` row-major `[n_mels, n_frames]` (n_frames = 3000 per 30 s chunk), matching HF `input_features` per chunk.

- [ ] **Step 1: Vendor fft, write mel header**

```bash
cp /home/mudler/_git/parakeet.cpp/src/fft.cpp src/fft.cpp
cp /home/mudler/_git/parakeet.cpp/src/fft.hpp src/fft.hpp
```
Adapt namespace to `mt`. `src/mel.hpp`:
```cpp
#pragma once
#include <vector>
#include "model_loader.hpp"
namespace mt {
class WhisperMel {
public:
    explicit WhisperMel(const ModelLoader& m);
    void compute(const std::vector<float>& samples, std::vector<float>& out,
                 int& n_mels, int& n_frames) const;
private:
    int n_fft_, hop_, n_mels_, n_bins_, n_samples_, nb_max_frames_;
    std::vector<float> window_;      // periodic Hann, length n_fft
    std::vector<float> fb_;          // [n_mels * n_bins], row-major
};
} // namespace mt
```

- [ ] **Step 2: Write failing test `tests/test_mel.cpp`**

```cpp
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
```
Register `mt_add_test(test_mel)` and set `LABELS "model"`.

- [ ] **Step 3: Run to verify it fails**

Run:
```bash
cmake --build build -j
MTD_TEST_GGUF=models/moss-transcribe-f32.gguf \
MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf \
ctest --test-dir build -R test_mel --output-on-failure
```
Expected: FAIL (link error / unimplemented `WhisperMel`).

- [ ] **Step 4: Implement `src/mel.cpp` (exact Whisper mel)**

```cpp
#include "mel.hpp"
#include "fft.hpp"
#include <cmath>
namespace mt {
WhisperMel::WhisperMel(const ModelLoader& m) {
    const auto& c = m.config();
    n_fft_ = c.feat_n_fft; hop_ = c.feat_hop; n_mels_ = c.feat_size;
    n_bins_ = n_fft_/2 + 1; n_samples_ = c.feat_n_samples; nb_max_frames_ = c.feat_nb_max_frames;
    window_.resize(n_fft_);
    for (int n = 0; n < n_fft_; ++n)                       // periodic Hann
        window_[n] = 0.5f * (1.0f - std::cos(2.0*M_PI*n/(double)n_fft_));
    std::vector<float> mf; std::vector<int64_t> sh;
    // mel_filters stored (80,201) row-major
    ggml_tensor* t = m.tensor("mel_filters");
    // read via loader helper into fb_ (n_mels*n_bins)
    // fb_[mel*n_bins + bin]
    fb_.assign((size_t)n_mels_*n_bins_, 0.0f);
    // ... copy from t->data (f32) into fb_
}
void WhisperMel::compute(const std::vector<float>& samples, std::vector<float>& out,
                         int& n_mels, int& n_frames) const {
    n_mels = n_mels_; n_frames = nb_max_frames_;
    // center STFT: reflect-pad by n_fft/2 on both ends
    const int pad = n_fft_/2;
    std::vector<float> x(samples.size() + 2*pad);
    for (int i = 0; i < pad; ++i) x[i] = samples[pad - i];                 // reflect
    for (size_t i = 0; i < samples.size(); ++i) x[pad+i] = samples[i];
    for (int i = 0; i < pad; ++i) x[pad+samples.size()+i] = samples[samples.size()-2-i];
    out.assign((size_t)n_mels_*n_frames, 0.0f);
    std::vector<float> frame(n_fft_), re, im;
    for (int t = 0; t < n_frames; ++t) {
        const int start = t*hop_;
        for (int n = 0; n < n_fft_; ++n) frame[n] = x[start+n]*window_[n];
        rfft(frame, re, im);                                              // n_bins bins
        for (int mbin = 0; mbin < n_mels_; ++mbin) {
            double acc = 0.0;
            const float* fbrow = &fb_[(size_t)mbin*n_bins_];
            for (int b = 0; b < n_bins_; ++b) {
                double power = (double)re[b]*re[b] + (double)im[b]*im[b];  // |stft|^2
                acc += (double)fbrow[b]*power;
            }
            out[(size_t)mbin*n_frames + t] = (float)std::log10(std::max(acc, 1e-10));
        }
    }
    // per-chunk normalization: x = max(x, max-8); x = (x+4)/4
    float mx = out[0];
    for (float v : out) mx = std::max(mx, v);
    const float floor = mx - 8.0f;
    for (float& v : out) { v = std::max(v, floor); v = (v + 4.0f)/4.0f; }
}
} // namespace mt
```
(Implement the `mel_filters` copy in the ctor via the loader's f32 tensor data; handle both stored shapes `[80,201]` and `[201,80]` by checking `t->ne`.)

- [ ] **Step 5: Add to MT_SRC, build, run the gate**

Add `src/fft.cpp src/mel.cpp` to `MT_SRC`. Run:
```bash
cmake --build build -j
MTD_TEST_GGUF=models/moss-transcribe-f32.gguf \
MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf \
ctest --test-dir build -R test_mel --output-on-failure
```
Expected: PASS — `compare` prints cosine ≥ 0.99999, max-abs small.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "M1: Whisper log-mel front-end (parity vs input_features)"
```

---

### Task 7: whisper_encoder — conv stem + 24 layers + final LN (M2)

**Files:**
- Create: `src/whisper_encoder.hpp`, `src/whisper_encoder.cpp`, `tests/test_whisper_encoder.cpp`
- Modify: `CMakeLists.txt` (`MT_SRC += whisper_encoder.cpp`), `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `mt::ModelLoader` (tensors `enc.*`, config `audio.*`), `mt::backend()`, `mt::compute_graph_with_inputs`.
- Produces: `class mt::WhisperEncoder { explicit WhisperEncoder(mt::ModelLoader&); /* loads enc.* */ void encode(const std::vector<float>& mel, int n_mels, int n_frames, std::vector<float>& out, int& out_T, int& out_D) const; };` — input mel `[n_mels, n_frames]` (one 30 s chunk, n_frames=3000), output `[out_D=1024, out_T=1500]` row-major (feature-fastest, matching ggml `ne=[1024,1500]`).

- [ ] **Step 1: Write encoder header + weight struct**

`src/whisper_encoder.hpp`:
```cpp
#pragma once
#include <vector>
#include "model_loader.hpp"
#include <ggml.h>
namespace mt {
struct WhisperLayer {
    ggml_tensor *attn_ln_w, *attn_ln_b;
    ggml_tensor *q_w, *q_b, *k_w, *v_w, *v_b, *o_w, *o_b;   // k has NO bias
    ggml_tensor *ffn_ln_w, *ffn_ln_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};
class WhisperEncoder {
public:
    explicit WhisperEncoder(ModelLoader& m);
    void encode(const std::vector<float>& mel, int n_mels, int n_frames,
                std::vector<float>& out, int& out_T, int& out_D) const;
private:
    int d_model_, n_layers_, n_heads_, ffn_, max_src_pos_;
    ggml_tensor *conv1_w_, *conv1_b_, *conv2_w_, *conv2_b_, *pos_embd_, *ln_post_w_, *ln_post_b_;
    std::vector<WhisperLayer> layers_;
};
} // namespace mt
```

- [ ] **Step 2: Write failing test `tests/test_whisper_encoder.cpp`**

```cpp
#include "whisper_encoder.hpp"
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
    // feed the dumped input_features (first chunk), not our own mel -> isolate the encoder
    std::vector<float> feat; std::vector<int64_t> sh;
    if (!mttest::load_baseline(base, "input_features", feat, sh)) return 1; // [1,80,3000]
    const int n_mels = (int)sh[1], n_frames = (int)sh[2];
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
```
Register `mt_add_test(test_whisper_encoder)` + `LABELS "model"`. (Note in a comment: ggml `ne=[D,T]` column-major storage has D contiguous, identical byte order to torch `[T,D]` row-major with D contiguous — so the flat buffers line up.)

- [ ] **Step 3: Run to verify it fails**

Run:
```bash
cmake --build build -j 2>&1 | tail -5
```
Expected: link error (unimplemented `WhisperEncoder`).

- [ ] **Step 4: Implement `src/whisper_encoder.cpp`**

Constructor loads all `enc.*` tensors by name via `m.tensor(...)` and `m.config()`. `encode()` builds one ggml graph (no_alloc) and runs it via `mt::compute_graph_with_inputs`. Graph (exact Whisper-Medium encoder):
```cpp
// mel input: ggml_new_tensor_2d(ctx, F32, n_frames, n_mels)  // ne=[3000,80]
// conv1: ggml_conv_1d_ph(ctx, conv1_w_, mel, /*s*/1, /*d*/1) -> [n_frames,1024]; + conv1_b_ broadcast; gelu_erf
// conv2: ggml_conv_1d_ph(ctx, conv2_w_, cur, /*s*/2, /*d*/1) -> [1500,1024]; + conv2_b_; gelu_erf
// cur -> transpose to [d_model, T]=[1024,1500]; add pos_embd_ (ne=[1024,1500])
// for each layer:
//   residual = cur
//   x = layer_norm(cur, attn_ln_w, attn_ln_b, 1e-5)          // Whisper LN eps 1e-5
//   q = linear(q_w,q_b,x); k = linear(k_w,nullptr,x); v = linear(v_w,v_b,x)
//   reshape q,k,v -> [head_dim=64, n_heads=16, T]; permute to [head_dim, T, n_heads]
//   scores = mul_mat(k,q); ggml_mul_mat_set_prec(scores, GGML_PREC_F32)
//   attn = soft_max_ext(scores, /*mask*/nullptr, scale=1/sqrt(64), 0)   // no causal mask (encoder)
//   ctx_ = mul_mat(cont(transpose(v)), attn); merge heads -> [1024,T]
//   x = linear(o_w,o_b, ctx_); cur = add(residual, x)
//   residual2 = cur
//   x = layer_norm(cur, ffn_ln_w, ffn_ln_b, 1e-5)
//   x = linear(fc2_w,fc2_b, gelu_erf(linear(fc1_w,fc1_b, x)))
//   cur = add(residual2, x)
// cur = layer_norm(cur, ln_post_w, ln_post_b, 1e-5)   // final LN
// capture cur as output [1024,1500]
```
Use `ggml_gelu_erf` (NOT `ggml_gelu`) for parity. Whisper attention scaling is `1/sqrt(head_dim)` via `soft_max_ext` scale (HF multiplies q by `scaling`; equivalent). Read `out` back with `mt::read_tensor_f32`.

- [ ] **Step 5: Add to MT_SRC, build, run the gate**

Run:
```bash
cmake --build build -j
MTD_TEST_GGUF=models/moss-transcribe-f32.gguf \
MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf \
ctest --test-dir build -R test_whisper_encoder --output-on-failure
```
Expected: PASS (cosine ≥ 0.99999; atol 5e-2 accommodates 24-layer accumulation). If FAIL, bisect: temporarily capture and compare the post-conv-stem tensor and layer-0 output (add a `baseline` dump for one intermediate if needed).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "M2: Whisper encoder (parity vs encoder_hidden)"
```

---

### Task 8: audio_adaptor — 4× time-merge + VQAdaptor (M3)

**Files:**
- Create: `src/audio_adaptor.hpp`, `src/audio_adaptor.cpp`, `tests/test_audio_adaptor.cpp`
- Modify: `CMakeLists.txt` (`MT_SRC += audio_adaptor.cpp`), `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `mt::ModelLoader` (tensors `adaptor.*`, config `audio_merge_size`, `adaptor_input_dim`, `adaptor_norm_eps`, `audio.d_model`), `mt::backend()`.
- Produces: `class mt::AudioAdaptor { explicit AudioAdaptor(ModelLoader&); void apply(const std::vector<float>& enc, int T, int D, std::vector<float>& out, int& N, int& H) const; };` — input concatenated+trimmed encoder output `[D=1024, T]`, output audio embeds `[H=1024, N=T/4]`.

- [ ] **Step 1: Write header**

```cpp
#pragma once
#include <vector>
#include "model_loader.hpp"
#include <ggml.h>
namespace mt {
class AudioAdaptor {
public:
    explicit AudioAdaptor(ModelLoader& m);
    void apply(const std::vector<float>& enc, int T, int D,
               std::vector<float>& out, int& N, int& H) const;
private:
    int merge_, in_dim_, hidden_; float eps_;
    ggml_tensor *fc1_w_, *fc1_b_, *fc2_w_, *fc2_b_, *ln_w_, *ln_b_;
};
} // namespace mt
```

- [ ] **Step 2: Write failing test `tests/test_audio_adaptor.cpp`**

```cpp
#include "audio_adaptor.hpp"
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
    // feed dumped encoder_hidden (single chunk, already trimmed to token_len*4 == 1500 for short.wav)
    std::vector<float> enc; std::vector<int64_t> sh;
    if (!mttest::load_baseline(base, "encoder_hidden", enc, sh)) return 1; // [1,1500,1024]
    const int T = (int)sh[1], D = (int)sh[2];
    mt::AudioAdaptor ad(m);
    std::vector<float> got; int N=0, H=0;
    ad.apply(enc, T, D, got, N, H);                       // [1024, 375]
    std::vector<float> ref; std::vector<int64_t> rsh;
    if (!mttest::load_baseline(base, "audio_embeds", ref, rsh)) return 1; // [375,1024]
    bool ok = mttest::compare(got, ref, "audio_adaptor", 1e-3f, 1e-2f);
    return ok ? 0 : 1;
}
```
Register `mt_add_test(test_audio_adaptor)` + `LABELS "model"`. (For short.wav the single chunk's `token_len*4` equals 1500, so no trimming needed in this isolated test; trimming/concat across chunks is exercised end-to-end in Phase 2 with `long.wav`.)

- [ ] **Step 3: Run to verify it fails**

Run: `cmake --build build -j 2>&1 | tail -5` — Expected: link error (unimplemented `AudioAdaptor`).

- [ ] **Step 4: Implement `src/audio_adaptor.cpp`**

```cpp
#include "audio_adaptor.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cmath>
namespace mt {
AudioAdaptor::AudioAdaptor(ModelLoader& m) {
    const auto& c = m.config();
    merge_ = c.audio_merge_size; in_dim_ = c.adaptor_input_dim;
    hidden_ = c.text_hidden; eps_ = c.adaptor_norm_eps;
    fc1_w_ = m.tensor("adaptor.fc1.w"); fc1_b_ = m.tensor("adaptor.fc1.b");
    fc2_w_ = m.tensor("adaptor.fc2.w"); fc2_b_ = m.tensor("adaptor.fc2.b");
    ln_w_  = m.tensor("adaptor.ln.w");  ln_b_  = m.tensor("adaptor.ln.b");
}
void AudioAdaptor::apply(const std::vector<float>& enc, int T, int D,
                         std::vector<float>& out, int& N, int& H) const {
    const int Ttrim = (T/merge_)*merge_;
    N = Ttrim/merge_; H = hidden_;
    // build graph: input enc as ne=[D, T] (D fastest). trim to [D,Ttrim], reshape to [D*merge, N]
    // NOTE reshape [D,Ttrim] -> [D*merge, N] groups merge consecutive frames' features (torch reshape parity).
    // fc1: linear(fc1_w,fc1_b) -> [hidden,N]; silu; fc2: linear(fc2_w,fc2_b) -> [hidden,N];
    // layer_norm(x, ln_w, ln_b, eps) over hidden (ne0). capture -> out [hidden,N].
    // (build via ggml graph + mt::compute_graph_with_inputs; read out via mt::read_tensor_f32)
}
} // namespace mt
```
Implement the graph exactly: `ggml_reshape_2d(enc_trim, D*merge, N)` → `mt::linear(fc1)` → `ggml_silu` → `mt::linear(fc2)` → `mt::layer_norm(., ln_w_, ln_b_, eps_)`. Trimming: view `[D, Ttrim]` of `[D,T]`.

- [ ] **Step 5: Add to MT_SRC, build, run the gate**

Run:
```bash
cmake --build build -j
MTD_TEST_GGUF=models/moss-transcribe-f32.gguf \
MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf \
ctest --test-dir build -R test_audio_adaptor --output-on-failure
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "M3: audio adaptor (time-merge + VQAdaptor, parity vs audio_embeds)"
```

---

### Task 9: Phase 1 end-to-end audio-path check + docs

Chain `audio_io → mel → whisper_encoder → audio_adaptor` from a raw wav and confirm it matches `audio_embeds`, proving the whole front end composes (short.wav single-chunk; multi-chunk handled in Phase 2).

**Files:**
- Create: `tests/test_audio_path_e2e.cpp`, `docs/conversion.md`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: all Phase 1 components. Produces: nothing new (integration test).

- [ ] **Step 1: Write the integration test**

```cpp
#include "audio_io.hpp"
#include "mel.hpp"
#include "whisper_encoder.hpp"
#include "audio_adaptor.hpp"
#include "parity.hpp"
#include <cstdlib>
int main() {
    const char* gguf = std::getenv("MTD_TEST_GGUF");
    const char* base = std::getenv("MTD_TEST_BASELINE");
    if (!gguf || !base) return 77;
    mt::ModelLoader m; if (!m.load(gguf)) return 1; m.promote_small_f16_to_f32();
    mt::Audio a; if (!mt::load_audio_16k_mono("tests/fixtures/short.wav", a)) return 1;
    a.samples.resize(m.config().feat_n_samples, 0.0f);           // single 30s chunk
    mt::WhisperMel mel(m); std::vector<float> feat; int nm=0,T=0; mel.compute(a.samples, feat, nm, T);
    mt::WhisperEncoder enc(m); std::vector<float> eh; int eT=0,eD=0; enc.encode(feat, nm, T, eh, eT, eD);
    mt::AudioAdaptor ad(m); std::vector<float> emb; int N=0,H=0; ad.apply(eh, eT, eD, emb, N, H);
    std::vector<float> ref; std::vector<int64_t> sh;
    if (!mttest::load_baseline(base, "audio_embeds", ref, sh)) return 1;
    // slightly looser: mel+encoder+adaptor accumulation
    return mttest::compare(emb, ref, "audio_path_e2e", 5e-2f, 5e-2f) ? 0 : 1;
}
```
Register `mt_add_test(test_audio_path_e2e)` + `LABELS "model"`.

- [ ] **Step 2: Build + run full model-labelled suite**

Run:
```bash
cmake --build build -j
MTD_TEST_GGUF=models/moss-transcribe-f32.gguf \
MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf \
ctest --test-dir build -L model --output-on-failure
```
Expected: `test_mel`, `test_whisper_encoder`, `test_audio_adaptor`, `test_audio_path_e2e` all PASS; `test_audio_io` PASS (unlabelled).

- [ ] **Step 3: Write `docs/conversion.md`**

Document: the GGUF tensor-name mapping (spec §3), the `mtd.*` KV list, the mel-filter storage convention, and the axis-order note (ggml `ne=[D,T]` D-fastest ↔ torch `[T,D]` row-major D-fastest). One paragraph per topic.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "M3: front-end end-to-end parity + conversion docs"
```

---

## Self-Review

**Spec coverage (Phase 1 scope = M0–M3):**
- M0 scaffold/converter/loader/info → Tasks 1–3 ✓
- M1 mel (+audio_io, parity harness) → Tasks 4–6 ✓
- M2 whisper_encoder → Task 7 ✓
- M3 audio_adaptor → Task 8; front-end composition → Task 9 ✓
- GGUF schema §3 (metadata, tokenizer, tensor names, mel_filters) → Task 3 ✓
- Parity harness §4 (baseline.gguf, tolerances, isolated inputs) → Task 4 + each gate ✓
- Deferred to later phases (correctly out of Phase 1): tokenizer runtime, audio_span/time-markers, qwen3 decode, fusion, transcript_parser, subtitle, CLI transcribe, C-API, F16/GPU, quant, LocalAI. These are M4–M10 and will get their own plans.

**Placeholder scan:** Graph-building bodies for `whisper_encoder`/`audio_adaptor` are given as exact op sequences with the parity-critical specifics (gelu_erf, no causal mask, k_proj no bias, `GGML_PREC_F32`, LN eps 1e-5 encoder / adaptor eps from config, reshape grouping). Vendored files (audio_io, fft, common, backend, model_loader, parity.hpp, ggml_extend) are copied verbatim with named adaptations. The converter's `rename()` and `add_tokenizer` token_type/merges are specified by reference to the exact spec §3 mapping and moss-tts `convert_tokenizer.py` — implementer fills every entry mechanically; the `check_convert.py` gate enforces completeness. No "TBD"/"handle edge cases" left.

**Type consistency:** `mt::ModelLoader`, `mt::Config`, `mt::WhisperMel`, `mt::WhisperEncoder::encode(...)`, `mt::AudioAdaptor::apply(...)`, `mttest::load_baseline/compare` signatures are identical across the tasks that define and consume them. Tensor names (`enc.*`, `adaptor.*`, `mel_filters`, `input_features`, `encoder_hidden`, `audio_embeds`) match between converter (Task 3), baseline (Task 4), and every test.

## Phase boundary

Phase 1 delivers a proven audio-embed front end. **Phase 2 (M4–M5)** — Qwen2 BPE tokenizer + audio-span/time-marker builder (bit-exact `input_ids`) + vendored Qwen3 decoder + masked_scatter fusion + greedy decode (bit-exact `generated_ids`) — is the next plan, written after Phase 1 lands and the shapes here are confirmed against the real dumps.
