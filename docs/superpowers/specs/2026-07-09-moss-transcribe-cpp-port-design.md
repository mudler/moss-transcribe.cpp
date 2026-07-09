# moss-transcribe.cpp — Design Spec

**Date:** 2026-07-09
**Status:** Approved (brainstorming), pending implementation plan
**Goal:** A loyal, numerically-faithful C++/ggml port of
[OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
(0.9B): dependency-free local inference (no Python/PyTorch/CUDA toolkit), one
self-contained GGUF, shipped as a full parakeet.cpp-style package (standalone
CLI + flat C-API + GGUF converter + LocalAI backend).

## 1. Reference summary

MOSS-Transcribe-Diarize is a **multimodal audio-LLM** that performs joint
long-form transcription + speaker diarization + timestamping in a single
autoregressive text generation. It is **not** a CTC/transducer ASR.

Grounded against two references:
- HF remote code: `moss_transcribe_diarize/{modeling,configuration,processing,inference_utils,transcript_parser,subtitle/*}.py`
- vLLM: `vllm/model_executor/models/moss_transcribe_diarize.py` (PR vllm-project/vllm#47729)

Both agree on every parity-relevant detail below.

### 1.1 Architecture (683 tensors, ~1.82 GB bf16)

```
log-mel (80×3000, 30s) ──> WhisperEncoder (Whisper-Medium, 24 layers) ──> (n_chunk,1500,1024)
        trim each chunk to token_len×4, concat along time ──────────────> (1, T, 1024)
        4× time-merge (reshape) ──────────────────────────────────────> (1, T/4, 4096)
        VQAdaptor (Linear 4096→1024 → SiLU → Linear 1024→1024 → LayerNorm) ─> (1, N, 1024)  = audio embeds
                                                                          │
text prompt ──> Qwen2 BPE ──> audio-span builder (time markers) ──> input_ids ──> embed_tokens ──> (1, seq, 1024)
        masked_scatter: audio embeds replace positions where input_ids == audio_token_id (151671)
                                                                          │
        Qwen3-0.6B decoder (28 layers) ──> final RMSNorm ──> lm_head (tied) ──> logits
        greedy decode to EOS (151645) ──> detokenize ──> "[start][Sxx]text[end]…" transcript
```

**Diarization/speaker labels are emitted inline as generated text tokens.**
There is no separate diarization/speaker-ID network. `[Sxx]` are relative,
per-recording labels assigned in order of appearance. No voiceprint/enrollment.
Output post-processing in the reference is just `text.strip()`.

### 1.2 Exact constants

**Text (Qwen3):** vocab 151936, hidden 1024, intermediate 3072, 28 layers,
16 Q heads, 8 KV heads, head_dim 128 (note Q proj → 16×128=2048 ≠ hidden),
Qwen3 per-head `q_norm`/`k_norm` (RMSNorm over head_dim), SwiGLU MLP,
RMSNorm eps 1e-6, rope_theta 1e6, max_pos 131072, `attention_bias=false`,
tied `lm_head`=`embed_tokens`. Plain sequential RoPE positions (no MRoPE).

**Audio (Whisper-Medium encoder):** num_mel_bins 80, d_model 1024, 24 layers,
16 heads, ffn 4096, max_source_positions 1500, activation gelu,
`scale_embedding=false`. Per-layer: pre-LN self-attn (q_proj+bias, **k_proj no
bias**, v_proj+bias, out_proj+bias), pre-LN FFN (fc1/fc2 with bias, gelu).
Conv stem: conv1 (80→1024, k3 s1 p1) → gelu → conv2 (1024→1024, k3 **s2** p1) →
gelu; learned `embed_positions` (1500×1024) added; final `layer_norm`.
Note: vLLM synthesizes a **zero bias** for k_proj — in ggml k_proj simply has
no bias term.

**Bridge:** audio_token_id 151671, audio_merge_size 4, adaptor_input_dim 4096
(= d_model × merge), adaptor LayerNorm eps = text rms_norm_eps (1e-6).

**Mel frontend (WhisperFeatureExtractor):** sampling_rate 16000, n_fft 400,
hop_length 160, feature_size 80, chunk_length 30s, n_samples 480000,
nb_max_frames 3000, dither 0.0, padding_value 0.0. Standard Whisper log-mel.

**Audio-span / time markers (parity-critical):** audio_tokens_per_second 12.5,
time_marker_every_seconds 2, enable_time_marker true.

**Generation:** eos_token_id 151645 (`<|im_end|>`), pad 151643, default
max_new_tokens 5120, greedy (do_sample=false).

### 1.3 Chunking & token-length math

Audio is split into 30s (480000-sample) chunks. Each chunk is padded to 30s for
the feature extractor, but its true token length is
`token_len = (chunk_samples − 1) // stride + 1` where
`stride = hop_length(160) × WHISPER_ENCODER_STRIDE(2) × audio_merge_size(4) = 1280`.
A full chunk → 375 tokens (= 1500 encoder frames / 4). The encoder output for
each chunk is trimmed to `token_len × 4` frames, all chunks concatenated along
time, then time-merged. Last partial chunk contributes fewer tokens.

### 1.4 The time-marker interleaver (`_audio_span_ids`)

The audio placeholder span is **not** N copies of `<|audio_pad|>`. Digit tokens
encoding elapsed whole seconds are interleaved every 2 s. Reference logic:

```
tokens_per_marker = int(audio_tokens_per_second × time_marker_every_seconds)  # 25
duration = audio_seq_len / audio_tokens_per_second
consumed = 0; output = []
for sec in range(2, int(duration)+1, 2):
    pos = (sec // 2) × tokens_per_marker
    seg = pos − consumed
    if seg > 0: output += [audio_token_id]×seg; consumed += seg
    output += [digit_token_id(d) for d in str(sec)]     # e.g. "12" → two digit tokens
remainder = audio_seq_len − consumed
if remainder > 0: output += [audio_token_id]×remainder
```

Total `audio_token_id` count == `audio_seq_len` (digit tokens are *added*, not
substituted), so masked_scatter fills exactly the audio positions. The full
placeholder expansion (from the chat template `<|audio_pad|>`) is
`[audio_start_id] + _audio_span_ids(N) + [audio_end_id]`.

### 1.5 Prompt (chat template, fixed)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|><|audio_end|>
{prompt}<|im_end|>
<|im_start|>assistant
```

`{prompt}` defaults to the Chinese transcribe+diarize instruction (stored in
GGUF as `mtd.default_prompt`). The single `<|audio_pad|>` is expanded per §1.4.

## 2. Repository layout

Standalone repo `moss-transcribe.cpp` (NOT merged into moss-tts.cpp — house
pattern is one-repo-per-model-port). Vendors `qwen3`, the BPE tokenizer, and
`ggml_extend.hpp` from moss-tts.cpp by copy-and-adapt; ggml as a submodule.

```
moss-transcribe.cpp/
├── CMakeLists.txt, Dockerfile, README.md, LICENSE
├── third_party/ggml/                    # submodule, pinned commit SHA
├── include/
│   ├── moss_transcribe.h                # C++ API
│   └── moss_transcribe_capi.h           # flat C ABI
├── src/
│   ├── common.{hpp,cpp}, ggml_extend.hpp
│   ├── backend.{hpp,cpp}                # ggml backend + persistent gallocr
│   ├── model_loader.{hpp,cpp}           # GGUF → Config + weights (all dims from metadata)
│   ├── model.{hpp,cpp}                  # Config + weight handles + forward-graph assembly
│   ├── audio_io.{hpp,cpp}               # decode → 16 kHz mono f32
│   ├── fft.{hpp,cpp}, mel.{hpp,cpp}, mel_gpu.{hpp,cpp}
│   ├── whisper_encoder.{hpp,cpp}
│   ├── audio_adaptor.{hpp,cpp}          # time-merge + VQAdaptor
│   ├── qwen3.{hpp,cpp}                  # vendored+adapted decoder
│   ├── tokenizer.{hpp,cpp}              # Qwen2 BPE + special/digit ids
│   ├── audio_span.{hpp,cpp}             # chat template + time-marker interleaver (pure int logic)
│   ├── generate.{hpp,cpp}               # embed → inject → greedy decode
│   ├── transcript_parser.{hpp,cpp}      # port of TranscriptStreamParser
│   ├── subtitle.{hpp,cpp}               # normalize_segments + SRT/ASS/JSON export
│   ├── moss_transcribe.cpp              # library entry (pipeline orchestration)
│   ├── moss_transcribe_capi.cpp
│   └── cli.cpp                          # transcribe / bench / info / stream
├── scripts/
│   ├── convert_moss_transcribe_to_gguf.py
│   ├── gen_baseline.py                  # dumps gold parity tensors
│   ├── quantize.*, publish_hf.py, requirements.txt
└── tests/                               # per-component parity + parser unit tests
```

**Isolation invariants:** `audio_span` is pure integer logic (no ggml) so the
time-marker interleaver is exhaustively unit-testable. `whisper_encoder` and
`qwen3` never reference each other — they meet only through the `(N,1024)` embed
tensor, so each is gated alone.

## 3. GGUF schema (metadata-driven, self-contained)

`general.architecture = "moss_transcribe_diarize"`. Everything the loader needs
is in the file; no sidecar config/vocab.

**Metadata KV (namespaced `mtd.`):**

| Group | Keys |
|---|---|
| Text | `text.{vocab_size, hidden, ffn, n_layers, n_heads, n_kv_heads, head_dim, rms_eps, rope_theta, max_pos, tied}` |
| Audio | `audio.{mel_bins, d_model, n_layers, n_heads, ffn, max_src_pos, act, scale_embed}` |
| Bridge | `audio_token_id, audio_merge_size, adaptor_input_dim, adaptor_norm_eps` |
| Mel | `feat.{sr, n_fft, hop, n_samples, nb_max_frames, feature_size, dither}` |
| Audio-span | `audio_tokens_per_second, time_marker_every_seconds, enable_time_marker` |
| Generation | `eos_token_id, default_max_new_tokens, default_prompt` |

**Tokenizer (fully embedded):** `tokenizer.ggml.model=gpt2` (Qwen2 BPE),
`.tokens`, `.merges`, `.pre="qwen2"`; ids: `eos=151645`, `pad=151643`,
`audio_start`, `audio_end`, `audio_pad=151671`; and
**`mtd.digit_token_ids` = [ids for "0".."9"]** stored explicitly (matches
`processor._get_digit_token_ids`) so the time-marker builder needs no runtime
BPE of digits.

**Tensors** (renamed to ggml idiom; f32 for the parity build; converter also
emits f16/q8_0/q6_k/q5_k/q4_k via `--outtype`):
- Encoder: `enc.conv1.{w,b}`, `enc.conv2.{w,b}`, `enc.pos_embd`, `enc.ln_post.{w,b}`,
  `enc.blk.{i}.{attn_q(w,b), attn_k(w), attn_v(w,b), attn_out(w,b), attn_ln(w,b),
  ffn_ln(w,b), ffn_1(w,b), ffn_2(w,b)}`
- Adaptor: `adaptor.fc1.{w,b}`, `adaptor.fc2.{w,b}`, `adaptor.ln.{w,b}`
- Decoder: `token_embd.w`, `blk.{i}.{attn_norm, attn_q, attn_k, attn_v, attn_out,
  attn_q_norm, attn_k_norm, ffn_norm, ffn_gate, ffn_up, ffn_down}`,
  `output_norm`, (`output` tied → alias `token_embd`)
- `mel_filters` (80×201) stored as a tensor so mel is reproducible without
  recomputing the filterbank.

## 4. Parity harness

`scripts/gen_baseline.py` loads the real model **fp32 on CPU** and dumps stage
boundaries for two fixtures: `short.wav` (<30 s, single chunk) and `long.wav`
(>30 s, multi-chunk — exercises chunk-concat + time markers). Reuse whisper.cpp
`samples/jfk.wav` for short; concatenate for long. Parity is numeric, so content
is irrelevant.

Output: raw little-endian `.bin` per tensor + `manifest.json`
(name → {shape, dtype, file}). The script calls each submodule directly
(`whisper_encoder`, `time_merge`, `vq_adaptor`, `inject_audio_features`), then
one `forward` (prefill logits) and one greedy `generate` (tokens).

**Gold tensors & gates:**

| # | Tensor | Gate |
|---|---|---|
| 1 | `input_features` (n_chunk,80,3000) | mel: cos ≥ 0.99999 |
| 2 | `audio_feature_lengths`, `audio_chunk_mapping` (int) | **bit-exact** |
| 3 | `input_ids` (expanded prompt) | **bit-exact** (both fixtures) |
| 4 | `encoder_hidden` (n_chunk,1500,1024) | cos ≥ 0.99999, max-abs bounded |
| 5 | `merged` (1,N,4096) | cos ≥ 0.99999 |
| 6 | `audio_embeds` (1,N,1024) | cos ≥ 0.99999 |
| 7 | `fused_embeds` (1,seq,1024) | cos ≥ 0.99999 |
| 8 | `lm_hidden` (1,seq,1024) | cos ≥ 0.99999 |
| 9 | `prompt_logits` (1,seq,vocab) | **argmax per position matches** |
| 10 | `generated_ids` (greedy, full) | **bit-exact — decisive gate** |
| 11 | `text` | sanity |

**Tolerances:** integer tensors (2,3,10) bit-exact; float tensors cosine
≥ 0.99999 AND max-abs-err ≤ small multiple of tensor std (f32 reduction-order
differences make bit-exact unrealistic through deep transformers). The
milestone-gating checks are #9 (argmax) and #10 (token ids) — robust to benign
float noise, unforgiving of real bugs. Each component test feeds the *dumped
input* for its stage (never the previous C++ stage's output) so errors are
isolated.

## 5. Milestone ladder

Each milestone = one parity gate + commit-and-push. Next doesn't start until the
current gate passes. Subagent per component where useful.

| M | Deliverable | Gate |
|---|---|---|
| M0 | Scaffold: repo, CMake, ggml submodule (pinned SHA), vendor qwen3/tokenizer/ggml_extend, converter + `gen_baseline.py` skeleton, fixtures, `info` subcommand | Builds; converter emits GGUF that loads + prints config |
| M1 | `audio_io` + `mel` (F32 CPU) | mel vs #1 |
| M2 | `whisper_encoder` | dumped `input_features` → vs #4 (per-layer if needed) |
| M3 | `audio_adaptor` (time-merge + VQAdaptor) | dumped encoder_hidden → vs #5, #6 |
| M4 | `tokenizer` + `audio_span` (time markers) | `input_ids` vs #3 **bit-exact** |
| M5 | `qwen3` + fusion + greedy `generate` | dumped `fused_embeds`→#8/#9; **wav→`generated_ids` vs #10 bit-exact** |
| M6 | Diarization/subtitle: `transcript_parser` + speaker-aware `normalize_segments` + SRT/ASS/JSON + speaker colors/names | 1:1 unit tests vs their `tests/test_subtitle_*` / `test_transcript_parser` |
| M7 | CLI (`transcribe`/`bench`/`info`/`stream`) + flat C-API + verbose_json segments | end-to-end CLI on fixtures; C-API smoke test |
| M8 | F16 + GPU backends (Metal/CUDA/Vulkan) | F16 token-id match; GPU == CPU |
| M9 | Quantization (q8_0/q6_k/q5_k/q4_k) + `publish_hf` | near-lossless @ q8, monotonic degradation |
| M10 | LocalAI backend (L0–L5): static-linked ggml `.so`, Go gRPC transcription (int64-ns timestamps), registration/gallery/tests/docs | LocalAI loads `.so` (`ldd` clean), transcribes |

M0–M7 is the full parakeet.cpp-style package; M8–M10 harden and ship. M6 (pure
string logic, depends only on M4's output format) may be built in parallel with
M2–M5.

## 6. Precision & backend scope

Parity gate and first CLI: **F32 on CPU** (tightest, most reproducible vs
reference dumps). Then F16 + enable Metal/CUDA/Vulkan (ggml handles them) once
numerically correct (M8). Quantization is M9.

## 7. Diarization / speaker handling (match reference exactly)

- Model emits `[start][Sxx]text[end]` inline; `transcript_parser` (ported from
  their regex-free char-scanning `TranscriptStreamParser`) yields
  `{start, end, speaker, text}`. `speaker` is the diarization result.
- `subtitle.normalize_segments` ports their speaker-aware pipeline:
  `_fix_overlaps` → `_merge_adjacent` (merges consecutive segments **only when
  speaker matches** and gap ≤ 0.3 s and combined length ≤ 2×max_chars) →
  `_split_long_segments` → `_fix_overlaps`. Defaults: min_duration 1.0,
  max_duration 6.0, max_chars 24, merge_gap 0.3.
- Export (SRT/ASS/JSON) carries speaker labels; ASS styling supports per-speaker
  colors and an optional `speaker_names` map (S01→"Alice"). This user-supplied
  naming is the full extent of "speaker identification" the reference offers.
- **No voiceprint/enrollment** — not in the reference, out of scope.

## 8. Out of scope

- The Python web/subtitle app, FFmpeg burn-in, FastAPI server, vLLM/SGLang
  runners (serving is LocalAI's job in M10).
- Enrolled/named speaker recognition (voiceprint) — a separate future spec if
  ever wanted.
- Training / fine-tuning.
- MOSS-Transcribe-Diarize **Pro** (not open-sourced).

## 9. Reusable sources

- `~/_git/moss-tts.cpp`: `qwen3.{cpp,hpp}`, BPE tokenizer, `ggml_extend.hpp`,
  CMake/backend patterns (vendor by copy-and-adapt).
- `~/_git/whisper.cpp`: Whisper encoder graph + mel frontend (read for
  reference; do not depend on — keep one self-contained GGUF).
- `~/_git/parakeet.cpp`: overall package shape — `model_loader`, `audio_io`,
  `backend`, GGUF converter, CLI, flat C-API, LocalAI backend layering.
- Reference model: HF remote code + vLLM PR #47729 (architecture + dump script).
