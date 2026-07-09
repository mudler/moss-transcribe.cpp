# GGUF conversion

`scripts/convert_moss_transcribe_to_gguf.py` turns the HF MOSS-Transcribe-Diarize
checkpoint into a single `moss-transcribe-*.gguf` that the C++ front end loads.
The `tests/python/check_convert.py` gate asserts that the produced tensor set
matches the expected set exactly (no missing / extra names). This document
records the conventions that conversion and the C++ loader must agree on.

## `mtd.*` metadata (KV) keys

All model hyper-parameters live under the `mtd.` prefix so they can be parsed
independently of tensor data (`src/model_loader.cpp`, defaults shown):

- Text / Qwen3 decoder: `mtd.text.vocab_size` (151936), `mtd.text.hidden` (1024),
  `mtd.text.ffn` (3072), `mtd.text.n_layers` (28), `mtd.text.n_heads` (16),
  `mtd.text.n_kv_heads` (8), `mtd.text.head_dim` (128), `mtd.text.rms_eps` (1e-6),
  `mtd.text.rope_theta` (1e6), `mtd.text.max_pos` (40960), `mtd.text.tied` (true).
- Audio / Whisper encoder: `mtd.audio.mel_bins` (80), `mtd.audio.d_model` (1024),
  `mtd.audio.n_layers` (24), `mtd.audio.n_heads` (16), `mtd.audio.ffn` (4096),
  `mtd.audio.max_src_pos` (1500), `mtd.audio.act` ("gelu"),
  `mtd.audio.scale_embed` (false).
- Bridge / adaptor: `mtd.audio_token_id` (151671), `mtd.audio_merge_size` (4),
  `mtd.adaptor_input_dim` (4096), `mtd.adaptor_norm_eps` (1e-6).
- Mel front end: `mtd.feat.sr` (16000), `mtd.feat.n_fft` (400), `mtd.feat.hop` (160),
  `mtd.feat.feature_size` (80), `mtd.feat.n_samples` (480000),
  `mtd.feat.nb_max_frames` (3000), `mtd.feat.dither` (0.0).
- Audio-span / time markers (Phase 2): `mtd.audio_tokens_per_second` (12.5),
  `mtd.time_marker_every_seconds` (2), `mtd.enable_time_marker` (true).
- Generation (Phase 2): `mtd.eos_token_id` (151645), `mtd.pad_token_id` (151643),
  `mtd.default_max_new_tokens` (5120), `mtd.default_prompt` (""),
  `mtd.digit_token_ids` (10 ids for "0".."9").

## HF → ggml tensor-name mapping

The converter's `rename()` maps HF parameter names to short ggml names:

- Whisper encoder (`enc.*`): `model.whisper_encoder.conv1.{weight,bias}` →
  `enc.conv1.{w,b}`, `conv2` → `enc.conv2.{w,b}`, `embed_positions.weight` →
  `enc.pos_embd`, `layer_norm.{weight,bias}` → `enc.ln_post.{w,b}`. Per-layer
  (`enc.blk.{i}.*`): `self_attn.{q,k,v,out}_proj` → `attn_{q,k,v,out}.w`
  (`k_proj` has no bias), `self_attn_layer_norm` → `attn_ln.{w,b}`,
  `final_layer_norm` → `ffn_ln.{w,b}`, `fc1`/`fc2` → `ffn_1.{w,b}`/`ffn_2.{w,b}`.
- VQ adaptor (`adaptor.*`): `model.vq_adaptor.layers.0` → `adaptor.fc1.{w,b}`,
  `layers.2` → `adaptor.fc2.{w,b}`, `layers.3` → `adaptor.ln.{w,b}`.
- Qwen3 decoder (`qwen3.blk.{i}.*.weight`): `input_layernorm` → `attn_norm`,
  `self_attn.{q,k,v,o}_proj` → `attn_{q,k,v,o}`, `self_attn.{q,k}_norm` →
  `attn_{q,k}_norm`, `post_attention_layernorm` → `ffn_norm`,
  `mlp.{gate,up,down}_proj` → `ffn_{gate,up,down}`. Plus the two top-level
  decoder tensors `token_embd.weight` (from `model.language_model.embed_tokens`)
  and `qwen3.output_norm.weight` (from `model.language_model.norm`).
- `lm_head.weight` is skipped: it is tied to `token_embd.weight`
  (`mtd.text.tied = true`), so the decoder reuses the embedding matrix.
- `mel_filters` (80, 201) is added separately (see below).

## Mel-filter storage convention

HF's feature extractor stores `mel_filters` as `(1 + n_fft/2, n_mels) = (201, 80)`.
The converter transposes it to `(80, 201)` (`np.ascontiguousarray(mel.T)`) so the
80 mel bins are the outer axis and the 201 FFT bins are contiguous, matching how
`src/mel.cpp` applies the filterbank (one contiguous 201-length row per mel bin).

## Axis-order convention

ggml stores tensors column-major with `ne = [D, T]` and the D axis contiguous
(fastest-varying). torch stores the same logical `[T, D]` tensor row-major, also
with D contiguous. Because both put D fastest, the flat float buffers are
byte-for-byte comparable with no transpose — every parity test compares the raw
C++ buffer against the dumped torch tensor directly (e.g. encoder output
`eh[t*1024 + d]`, feature-fastest). Dumped tensors with a leading batch/size-1
dim collapse under `ggml_n_dims`, so `load_baseline` reports the trailing dims.

## Encoder trimming rule

The Whisper encoder always runs on a padded 30 s chunk (`feat.n_samples`=480000
samples → mel `[80, 3000]` → encoder `[1024, 1500]`). Before the time-merge /
adaptor, each chunk is trimmed to `token_len * 4` frames, where the token length
comes from the *unpadded* sample count (this is the C++ equivalent of the
reference `get_audio_features` logic, not a baseline lookup):

```
stride    = feat.hop(160) * WHISPER_ENCODER_STRIDE(2) * audio_merge_size(4) = 1280
token_len = (n_unpadded_samples - 1) / stride + 1        # integer division
```

For jfk `short.wav` (~176000 samples): `token_len = (176000-1)/1280 + 1 = 138`,
so the encoder output is trimmed to `138*4 = 552` frames, and the adaptor emits
`N = 138` audio embeddings. Because the encoder buffer is feature-fastest, the
trim is just `eh.resize(552 * 1024)`. Phase-2 orchestration will generalize this
to multi-chunk audio (each chunk trimmed then concatenated). This whole chain is
exercised end-to-end by `tests/test_audio_path_e2e.cpp`.

## Known ggml workarounds (discovered in Phase 1)

Two upstream ggml limitations required custom code to reach numerical parity:

- **Direct DFT for the mel STFT** (`src/mel.cpp`): Whisper's `n_fft = 400` is not
  a power of two, so the radix-2 `mt::rfft` cannot be used. Mel computation uses
  precomputed cos/sin twiddle tables and a direct DFT over the 201 output bins.
- **F32 im2col conv1d** (`conv1d_ph_f32` in `src/whisper_encoder.cpp`): ggml's
  `ggml_conv_1d_ph` hardcodes an F16 im2col (requiring an F16 kernel), which
  would lose precision on the encoder's two conv layers. We use a custom F32
  im2col conv so the convolutions stay in F32 and match the reference.
