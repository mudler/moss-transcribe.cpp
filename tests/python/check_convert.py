#!/usr/bin/env python3
"""Gate for the moss-transcribe GGUF converter.

Asserts the produced GGUF carries the required `mtd.*`/tokenizer KV and the
expected tensor names (with the qwen3.* decoder naming from the task-3
interface notes), then prints "OK <n> tensors" and exits 0.
"""
import sys
import gguf

r = gguf.GGUFReader(sys.argv[1])
kv = {f.name: f for f in r.fields.values()}

need_kv = [
    "general.architecture", "mtd.text.hidden", "mtd.audio.d_model",
    "mtd.audio_token_id", "mtd.feat.n_fft", "mtd.digit_token_ids",
    "mtd.audio_tokens_per_second", "tokenizer.tokens",
]
for k in need_kv:
    assert k in kv, f"missing KV {k}"

names = {t.name for t in r.tensors}
need_t = [
    "mel_filters", "enc.conv1.w", "enc.conv2.w", "enc.pos_embd", "enc.ln_post.w",
    "enc.blk.0.attn_q.w", "enc.blk.23.ffn_2.w",
    "adaptor.fc1.w", "adaptor.fc2.w", "adaptor.ln.w",
    "token_embd.weight", "qwen3.blk.0.attn_q.weight",
    "qwen3.blk.27.ffn_down.weight", "qwen3.output_norm.weight",
]
for t in need_t:
    assert t in names, f"missing tensor {t}"

mf = next(t for t in r.tensors if t.name == "mel_filters")
assert list(mf.shape) == [201, 80] or list(mf.shape) == [80, 201], mf.shape

print("OK", len(names), "tensors")
