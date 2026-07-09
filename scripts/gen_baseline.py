#!/usr/bin/env python3
"""Dump GOLD reference tensors from the genuine MOSS-Transcribe-Diarize model.

Produces a baseline.gguf consumed by the C++ parity tests (tests/parity.hpp).
Every tensor here comes from the authors' real remote-code model + processor,
so downstream component gates compare against ground truth.

Reference pipeline loading (transformers 5.3 cannot AutoProcessor this model):
  * The model loads fine via AutoModelForCausalLM(trust_remote_code=True).
  * The processor is built DIRECTLY from the authors' MossTranscribeDiarizeProcessor
    class because AutoProcessor/AutoTokenizer trip over a `fix_mistral_regex`
    kwarg bug. We strip that single key from a symlinked copy of the genuine
    tokenizer files (nothing else changes) and load the genuine Qwen2 tokenizer,
    genuine WhisperFeatureExtractor, and genuine chat_template.jinja.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import torch
import gguf

from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor
from moss_transcribe_diarize.processing_moss_transcribe_diarize import (
    MossTranscribeDiarizeProcessor,
)
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    prepare_inputs,
)

AUDIO_TOKEN_ID = 151671  # <|audio_pad|>


def build_genuine_processor(model_dir: str) -> MossTranscribeDiarizeProcessor:
    """Construct the authors' processor, working around the fix_mistral_regex bug."""
    # Symlink the genuine tokenizer files into a temp dir, dropping only the
    # offending `fix_mistral_regex` key from tokenizer_config.json.
    patch_dir = tempfile.mkdtemp(prefix="moss_tok_")
    for fname in (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
    ):
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            os.symlink(os.path.abspath(src), os.path.join(patch_dir, fname))
    tok_cfg = json.load(open(os.path.join(model_dir, "tokenizer_config.json")))
    tok_cfg.pop("fix_mistral_regex", None)
    json.dump(tok_cfg, open(os.path.join(patch_dir, "tokenizer_config.json"), "w"))

    tokenizer = AutoTokenizer.from_pretrained(patch_dir)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    proc_cfg = json.load(open(os.path.join(model_dir, "processor_config.json")))
    chat_template = open(os.path.join(model_dir, "chat_template.jinja")).read()

    processor = MossTranscribeDiarizeProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        audio_tokens_per_second=proc_cfg["audio_tokens_per_second"],
        audio_merge_size=proc_cfg["audio_merge_size"],
        time_marker_every_seconds=proc_cfg["time_marker_every_seconds"],
        enable_time_marker=proc_cfg["enable_time_marker"],
        chat_template=chat_template,
    )
    return processor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("wav")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument(
        "--full-gold",
        action="store_true",
        help="also dump fused_embeds/lm_hidden/prompt_logits (short fixture).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="greedy decode length for generated_ids (0 = skip generate).",
    )
    args = ap.parse_args()

    device = torch.device("cpu")
    dtype = torch.float32

    print("loading genuine reference model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        dtype=dtype,
        device_map="cpu",
    ).eval()
    # The checkpoint ships only embed_tokens.weight (tie_word_embeddings=True),
    # but transformers 5.3 leaves lm_head.weight on the meta device untied.
    # Force the genuine tie so lm_head == embed_tokens (else logits are random).
    model.tie_weights()
    assert model.lm_head.weight.device.type != "meta", "lm_head still on meta after tie_weights()"
    assert model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr(), \
        "lm_head not tied to embed_tokens"
    print("building genuine reference processor ...", flush=True)
    proc = build_genuine_processor(args.model_dir)

    msgs = build_transcription_messages(args.wav)
    inputs = prepare_inputs(proc, msgs, device=device)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    writer = gguf.GGUFWriter(args.output, "moss_transcribe_diarize_baseline")

    def dump(name: str, t: torch.Tensor) -> None:
        if t.dtype.is_floating_point:
            a = t.detach().cpu().to(torch.float32).numpy()
        else:
            a = t.detach().cpu().to(torch.int32).numpy()
        writer.add_tensor(name, np.ascontiguousarray(a))
        print(f"  dumped {name} shape={tuple(a.shape)} dtype={a.dtype}", flush=True)

    mdl = model.model  # MossTranscribeDiarizeModel

    with torch.no_grad():
        # ---- always dumped (both fixtures) --------------------------------
        dump("input_features", inputs["input_features"])
        dump("audio_feature_lengths", inputs["audio_feature_lengths"])
        dump("audio_chunk_mapping", inputs["audio_chunk_mapping"])
        dump("input_ids", inputs["input_ids"][0])

        # UNtrimmed full encoder output (n_chunk, 1500, 1024)
        enc = mdl.whisper_encoder(
            inputs["input_features"].to(dtype), return_dict=True
        ).last_hidden_state
        dump("encoder_hidden", enc)

        # audio_embeds = trimmed + time-merged + VQAdaptor'd (N, 1024)
        feats = model.get_audio_features(
            inputs["input_features"],
            inputs["audio_feature_lengths"],
            inputs.get("audio_chunk_mapping"),
        )
        audio_embeds = torch.cat([f.squeeze(0) for f in feats], dim=0)
        dump("audio_embeds", audio_embeds)

        # merged = pre-adaptor time-merge output (N, D*merge). Replicate the
        # genuine trim+concat+time_merge per audio (matches get_audio_features).
        chunk_mapping = inputs.get("audio_chunk_mapping")
        if chunk_mapping is None:
            chunk_mapping = torch.zeros(
                inputs["input_features"].shape[0], dtype=torch.long
            )
        lengths = inputs["audio_feature_lengths"].tolist()
        num_audios = int(chunk_mapping.max().item()) + 1
        per_audio = [[] for _ in range(num_audios)]
        for chunk_idx, token_len in enumerate(lengths):
            sample_idx = int(chunk_mapping[chunk_idx].item())
            per_audio[sample_idx].append(enc[chunk_idx : chunk_idx + 1, : int(token_len) * 4])
        merged_parts = []
        for parts in per_audio:
            feat = torch.cat(parts, dim=1).to(mdl.dtype)
            merged_parts.append(mdl.time_merge(feat).squeeze(0))
        merged = torch.cat(merged_parts, dim=0)
        dump("merged", merged)

        # ---- full decode gold (short fixture) -----------------------------
        if args.full_gold:
            embeds = mdl.get_input_embeddings()(inputs["input_ids"])
            fused = mdl.inject_audio_features(
                inputs["input_ids"],
                embeds,
                inputs["input_features"],
                inputs["audio_feature_lengths"],
                inputs.get("audio_chunk_mapping"),
            )
            dump("fused_embeds", fused[0])

            out = model(
                inputs_embeds=fused,
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            dump("prompt_logits", out.logits[0])
            dump("lm_hidden", out.hidden_states[-1][0])

        # ---- greedy generation gold --------------------------------------
        text = None
        if args.max_new_tokens > 0:
            print(f"generating (greedy, max_new_tokens={args.max_new_tokens}) ...", flush=True)
            gen = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs.get("audio_chunk_mapping"),
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
            dump("generated_ids", gen[0])
            prompt_len = inputs["input_ids"].shape[1]
            text = proc.tokenizer.decode(
                gen[0][prompt_len:], skip_special_tokens=True
            ).strip()

    if text is not None:
        writer.add_string("baseline.text", text)
        print("baseline.text:", repr(text), flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
