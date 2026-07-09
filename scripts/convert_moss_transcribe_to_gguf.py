#!/usr/bin/env python3
"""Convert the HF MOSS-Transcribe-Diarize checkpoint into a self-contained
`mtd.*` GGUF for moss-transcribe.cpp.

Every tensor is stored contiguous, f32, in its native torch shape (NO transpose)
except `mel_filters`, which is transposed from HF (201, 80) to (80, 201).
`lm_head.weight` is skipped (tied to token embeddings).

The decoder tensors use the `qwen3.blk.{i}.*.weight` naming expected by the
Phase-2 vendored qwen3 loader (see task-3 interface notes section C).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import gguf
from transformers import AutoConfig, WhisperFeatureExtractor
from tokenizers import Tokenizer
from safetensors.torch import load_file

# Exact DEFAULT_PROMPT from moss_transcribe_diarize/inference_utils.py.
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


# --------------------------------------------------------------------------
# Quantization policy.
#
# Quantize ONLY the large 2-D weights fed directly into ggml_mul_mat (ggml
# dequantizes f16/q8_0 src0 on the fly). Everything else -- biases, norms, the
# conv stem, pos_embd, mel_filters -- stays F32 (excluded because it does not
# match the allowlist). `token_embd.weight` is quantizable: Q0 made the embed
# lookup dtype-agnostic (ggml_get_rows) and the tied lm_head goes through
# ggml_mul_mat. See .superpowers/sdd/q1-interface-notes.md.
# --------------------------------------------------------------------------
_QUANTIZABLE_PATTERNS = [
    r"^token_embd\.weight$",                          # embeddings + tied lm_head
    r"^qwen3\.blk\.\d+\.attn_[qkvo]\.weight$",         # decoder attention projections
    r"^qwen3\.blk\.\d+\.ffn_(gate|up|down)\.weight$",  # decoder SwiGLU
    r"^enc\.blk\.\d+\.attn_(q|k|v|out)\.w$",           # whisper encoder attention (.w only)
    r"^enc\.blk\.\d+\.ffn_(1|2)\.w$",                  # whisper encoder FFN
    r"^adaptor\.(fc1|fc2)\.w$",                        # VQAdaptor linears
]
_QRE = [re.compile(p) for p in _QUANTIZABLE_PATTERNS]


def should_quantize(name, ggml_ne, dtype):
    """Return the ggml quantization type for ``name`` given the requested dtype.

    ``ggml_ne`` is the reverse of the numpy shape, so ``ggml_ne[0]`` is the
    leading/contraction axis q8_0 blocks along (block size 32). Returns None
    (keep F32) unless the tensor is on the linear-weight allowlist, is at least
    2-D with both dims >= 32, and (for q8_0) has a block-aligned leading dim.
    """
    if dtype == "f32":
        return None
    if not any(rx.match(name) for rx in _QRE):
        return None
    if len(ggml_ne) < 2 or min(ggml_ne[0], ggml_ne[1]) < 32:
        return None
    if dtype == "f16":
        return gguf.GGMLQuantizationType.F16
    if dtype == "q8_0":
        if ggml_ne[0] % 32 != 0:
            return None
        return gguf.GGMLQuantizationType.Q8_0
    return None


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
def _rope_theta(t):
    if getattr(t, "rope_theta", None) is not None:
        return float(t.rope_theta)
    rp = getattr(t, "rope_parameters", None) or getattr(t, "rope_scaling", None)
    if rp and "rope_theta" in rp:
        return float(rp["rope_theta"])
    raise AttributeError("could not find rope_theta on text_config")


def add_metadata(w, cfg):
    # general.architecture is already set by the GGUFWriter constructor.
    t = cfg.text_config
    a = cfg.audio_config
    # Text (Qwen3).
    w.add_uint32("mtd.text.vocab_size", t.vocab_size)
    w.add_uint32("mtd.text.hidden", t.hidden_size)
    w.add_uint32("mtd.text.ffn", t.intermediate_size)
    w.add_uint32("mtd.text.n_layers", t.num_hidden_layers)
    w.add_uint32("mtd.text.n_heads", t.num_attention_heads)
    w.add_uint32("mtd.text.n_kv_heads", t.num_key_value_heads)
    w.add_uint32("mtd.text.head_dim", t.head_dim)
    w.add_float32("mtd.text.rms_eps", float(t.rms_norm_eps))
    w.add_float32("mtd.text.rope_theta", _rope_theta(t))
    w.add_uint32("mtd.text.max_pos", int(t.max_position_embeddings))
    w.add_bool("mtd.text.tied", True)
    # Audio (Whisper).
    w.add_uint32("mtd.audio.mel_bins", a.num_mel_bins)
    w.add_uint32("mtd.audio.d_model", a.d_model)
    w.add_uint32("mtd.audio.n_layers", a.encoder_layers)
    w.add_uint32("mtd.audio.n_heads", a.encoder_attention_heads)
    w.add_uint32("mtd.audio.ffn", a.encoder_ffn_dim)
    w.add_uint32("mtd.audio.max_src_pos", a.max_source_positions)
    w.add_string("mtd.audio.act", a.activation_function)
    w.add_bool("mtd.audio.scale_embed", bool(a.scale_embedding))
    # Bridge.
    w.add_uint32("mtd.audio_token_id", cfg.audio_token_id)
    w.add_uint32("mtd.audio_merge_size", cfg.audio_merge_size)
    w.add_uint32("mtd.adaptor_input_dim", cfg.adaptor_input_dim)
    w.add_float32("mtd.adaptor_norm_eps", float(t.rms_norm_eps))


def add_feature_extractor(w, fe, proc_cfg):
    w.add_uint32("mtd.feat.sr", fe.sampling_rate)
    w.add_uint32("mtd.feat.n_fft", fe.n_fft)
    w.add_uint32("mtd.feat.hop", fe.hop_length)
    w.add_uint32("mtd.feat.feature_size", fe.feature_size)
    w.add_uint32("mtd.feat.n_samples", fe.n_samples)
    w.add_uint32("mtd.feat.nb_max_frames", fe.nb_max_frames)
    w.add_float32("mtd.feat.dither", float(getattr(fe, "dither", 0.0)))
    w.add_float32("mtd.audio_tokens_per_second", float(proc_cfg["audio_tokens_per_second"]))
    w.add_uint32("mtd.time_marker_every_seconds", int(proc_cfg["time_marker_every_seconds"]))
    w.add_bool("mtd.enable_time_marker", bool(proc_cfg["enable_time_marker"]))
    # Slaney mel filters. HF fe.mel_filters is (1 + n_fft//2, n_mels) = (201, 80).
    # Store transposed to (80, 201), row-major f32.
    mel = np.asarray(fe.mel_filters, dtype=np.float32)
    assert mel.shape == (201, 80), f"unexpected mel_filters shape {mel.shape}"
    w.add_tensor("mel_filters", np.ascontiguousarray(mel.T))


def add_tokenizer(w, tk, model_dir):
    # Raw byte-level BPE tokenizer (Qwen2). We build the vocab / merges /
    # special-token KV directly from tokenizer.json so we do not depend on the
    # transformers tokenizer wrapper (which is incompatible with this repo's
    # `fix_mistral_regex` flag under transformers 5.x).
    tj_path = os.path.join(model_dir, "tokenizer.json")
    with open(tj_path, "r", encoding="utf-8") as fh:
        tj = json.load(fh)

    vocab = tk.get_vocab(with_added_tokens=True)  # token -> id
    id2tok = {i: t for t, i in vocab.items()}
    added = {int(a["id"]): a for a in tj.get("added_tokens", [])}
    n = max(max(id2tok), max(added) if added else -1) + 1

    tokens = []
    token_type = []
    for i in range(n):
        if i in added:
            tokens.append(added[i]["content"])
            token_type.append(1 if added[i]["special"] else 3)  # CONTROL / USER_DEFINED
        else:
            tokens.append(id2tok.get(i, ""))
            token_type.append(0)  # NORMAL

    w.add_array("tokenizer.tokens", tokens)
    w.add_array("tokenizer.token_type", [int(x) for x in token_type])

    # Merges (rank == index). tokenizer.json stores each as ["a", "b"] or "a b".
    merges = []
    for m in tj["model"]["merges"]:
        if isinstance(m, (list, tuple)):
            merges.append(f"{m[0]} {m[1]}")
        else:
            merges.append(m)
    w.add_array("tokenizer.merges", merges)

    # Special tokens: every added token marked special.
    special_ids = sorted(i for i, a in added.items() if a["special"])
    special_texts = [added[i]["content"] for i in special_ids]
    w.add_array("tokenizer.special_tokens_ids", [int(x) for x in special_ids])
    w.add_array("tokenizer.special_tokens_text", special_texts)

    w.add_int32("tokenizer.bos_id", -1)          # Qwen2 has no BOS
    w.add_int32("tokenizer.eos_id", 151645)      # <|im_end|>
    w.add_int32("tokenizer.pad_id", 151643)      # <|endoftext|>

    # Digit token ids (parity with processor._get_digit_token_ids()).
    digits = []
    for d in "0123456789":
        ids = tk.encode(d, add_special_tokens=False).ids
        assert len(ids) == 1, f"digit {d!r} is not a single token: {ids}"
        digits.append(int(ids[0]))
    w.add_array("mtd.digit_token_ids", digits)

    w.add_uint32("mtd.eos_token_id", 151645)
    w.add_uint32("mtd.pad_token_id", 151643)
    w.add_uint32("mtd.default_max_new_tokens", 5120)
    w.add_string("mtd.default_prompt", DEFAULT_PROMPT)


# --------------------------------------------------------------------------
# Tensor name mapping (HF state_dict -> ggml GGUF names).  See notes section C.
# --------------------------------------------------------------------------
_ENC_LAYER_MAP = {
    "self_attn.q_proj.weight":         "attn_q.w",
    "self_attn.q_proj.bias":           "attn_q.b",
    "self_attn.k_proj.weight":         "attn_k.w",   # no bias
    "self_attn.v_proj.weight":         "attn_v.w",
    "self_attn.v_proj.bias":           "attn_v.b",
    "self_attn.out_proj.weight":       "attn_out.w",
    "self_attn.out_proj.bias":         "attn_out.b",
    "self_attn_layer_norm.weight":     "attn_ln.w",
    "self_attn_layer_norm.bias":       "attn_ln.b",
    "final_layer_norm.weight":         "ffn_ln.w",
    "final_layer_norm.bias":           "ffn_ln.b",
    "fc1.weight":                      "ffn_1.w",
    "fc1.bias":                        "ffn_1.b",
    "fc2.weight":                      "ffn_2.w",
    "fc2.bias":                        "ffn_2.b",
}

_ENC_TOP_MAP = {
    "model.whisper_encoder.conv1.weight":            "enc.conv1.w",
    "model.whisper_encoder.conv1.bias":              "enc.conv1.b",
    "model.whisper_encoder.conv2.weight":            "enc.conv2.w",
    "model.whisper_encoder.conv2.bias":              "enc.conv2.b",
    "model.whisper_encoder.embed_positions.weight":  "enc.pos_embd",
    "model.whisper_encoder.layer_norm.weight":       "enc.ln_post.w",
    "model.whisper_encoder.layer_norm.bias":         "enc.ln_post.b",
}

_ADAPTOR_MAP = {
    "model.vq_adaptor.layers.0.weight": "adaptor.fc1.w",
    "model.vq_adaptor.layers.0.bias":   "adaptor.fc1.b",
    "model.vq_adaptor.layers.2.weight": "adaptor.fc2.w",
    "model.vq_adaptor.layers.2.bias":   "adaptor.fc2.b",
    "model.vq_adaptor.layers.3.weight": "adaptor.ln.w",
    "model.vq_adaptor.layers.3.bias":   "adaptor.ln.b",
}

_DEC_LAYER_MAP = {
    "input_layernorm.weight":          "attn_norm.weight",
    "self_attn.q_proj.weight":         "attn_q.weight",
    "self_attn.k_proj.weight":         "attn_k.weight",
    "self_attn.v_proj.weight":         "attn_v.weight",
    "self_attn.o_proj.weight":         "attn_o.weight",
    "self_attn.q_norm.weight":         "attn_q_norm.weight",
    "self_attn.k_norm.weight":         "attn_k_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight":            "ffn_gate.weight",
    "mlp.up_proj.weight":              "ffn_up.weight",
    "mlp.down_proj.weight":            "ffn_down.weight",
}

_DEC_TOP_MAP = {
    "model.language_model.embed_tokens.weight": "token_embd.weight",
    "model.language_model.norm.weight":         "qwen3.output_norm.weight",
}


def rename(name):
    """Map an HF state_dict tensor name to its ggml GGUF name, or None if the
    tensor should be dropped."""
    if name == "lm_head.weight":
        return None  # tied to token embeddings

    if name in _ENC_TOP_MAP:
        return _ENC_TOP_MAP[name]
    if name in _ADAPTOR_MAP:
        return _ADAPTOR_MAP[name]
    if name in _DEC_TOP_MAP:
        return _DEC_TOP_MAP[name]

    # Whisper encoder layers.
    prefix = "model.whisper_encoder.layers."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        idx, sub = rest.split(".", 1)
        if sub in _ENC_LAYER_MAP:
            return f"enc.blk.{int(idx)}.{_ENC_LAYER_MAP[sub]}"
        return f"__UNMAPPED__:{name}"

    # Qwen3 decoder layers.
    prefix = "model.language_model.layers."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        idx, sub = rest.split(".", 1)
        if sub in _DEC_LAYER_MAP:
            return f"qwen3.blk.{int(idx)}.{_DEC_LAYER_MAP[sub]}"
        return f"__UNMAPPED__:{name}"

    return f"__UNMAPPED__:{name}"


def expected_names(cfg):
    """Full set of ggml tensor names the converter must produce (excluding
    mel_filters, which is added separately)."""
    exp = set()
    exp.update(_ENC_TOP_MAP.values())
    exp.update(_ADAPTOR_MAP.values())
    exp.update(_DEC_TOP_MAP.values())
    for i in range(cfg.audio_config.encoder_layers):
        for v in _ENC_LAYER_MAP.values():
            exp.add(f"enc.blk.{i}.{v}")
    for i in range(cfg.text_config.num_hidden_layers):
        for v in _DEC_LAYER_MAP.values():
            exp.add(f"qwen3.blk.{i}.{v}")
    return exp


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f32")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    fe = WhisperFeatureExtractor.from_pretrained(args.model_dir)
    with open(os.path.join(args.model_dir, "processor_config.json"), "r", encoding="utf-8") as fh:
        proc_cfg = json.load(fh)
    tk = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))

    # Load all safetensors shards.
    sd = {}
    shards = sorted(glob.glob(os.path.join(args.model_dir, "model*.safetensors")))
    if not shards:
        shards = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
    assert shards, f"no safetensors shards found in {args.model_dir}"
    for shard in shards:
        sd.update(load_file(shard))
    print(f"loaded {len(sd)} tensors from {len(shards)} shard(s)")

    w = gguf.GGUFWriter(args.out, "moss_transcribe_diarize")
    add_metadata(w, cfg)
    add_feature_extractor(w, fe, proc_cfg)
    add_tokenizer(w, tk, args.model_dir)

    produced = set()
    unmapped = []
    quantized = 0
    for hf_name, tensor in sd.items():
        gname = rename(hf_name)
        if gname is None:
            continue  # deliberately skipped (lm_head)
        if gname.startswith("__UNMAPPED__:"):
            unmapped.append(hf_name)
            continue
        arr = np.ascontiguousarray(tensor.to(torch.float32).numpy())
        # ggml ne is the reverse of the numpy shape; ne[0] = leading axis.
        ggml_ne = list(arr.shape)[::-1]
        qtype = should_quantize(gname, ggml_ne, args.dtype)
        if qtype is None:
            w.add_tensor(gname, arr)  # F32
        else:
            raw = gguf.quantize(arr, qtype)
            # gguf derives the element shape from the raw byte shape + raw_dtype.
            w.add_tensor(gname, raw, raw_shape=raw.shape, raw_dtype=qtype)
            quantized += 1
        produced.add(gname)

    # ------------------------------------------------------------------
    # Loud assertions: nothing unmapped, nothing expected missing.
    # ------------------------------------------------------------------
    exp = expected_names(cfg)
    missing = sorted(exp - produced)
    extra = sorted(produced - exp)  # produced but not in expected set (should be empty)

    print("=== converter assertion ===")
    print(f"produced (weight tensors): {len(produced)}")
    print(f"expected (weight tensors): {len(exp)}")
    print(f"unmapped HF tensors:       {len(unmapped)}")
    print(f"missing expected names:    {len(missing)}")
    print(f"unexpected produced names: {len(extra)}")
    if unmapped:
        print("UNMAPPED:", unmapped, file=sys.stderr)
    if missing:
        print("MISSING:", missing, file=sys.stderr)
    if extra:
        print("EXTRA:", extra, file=sys.stderr)
    assert not unmapped, f"{len(unmapped)} HF tensor(s) did not map to any ggml name"
    assert not missing, f"{len(missing)} expected ggml name(s) never produced"
    assert not extra, f"{len(extra)} produced name(s) not in the expected set"
    print("assertion OK: all tensors mapped, all expected names produced")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} (+mel_filters, total {len(produced) + 1} tensors, "
          f"dtype={args.dtype}, quantized={quantized})")


if __name__ == "__main__":
    main()
