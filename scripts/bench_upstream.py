"""Benchmark the UPSTREAM PyTorch MOSS-Transcribe-Diarize on CPU, for an
apples-to-apples speed comparison vs moss-transcribe.cpp (same audio, same
thread budget, warm run). Uses the authors' genuine model + processor + the
reference inference path (inference_utils.generate_transcription).
"""

from __future__ import annotations

import argparse
import sys
import time

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, "scripts")
from gen_baseline import build_genuine_processor  # noqa: E402
from moss_transcribe_diarize.inference_utils import (  # noqa: E402
    build_transcription_messages,
    generate_transcription,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("wav")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    ap.add_argument("--max-new", type=int, default=5120)
    ap.add_argument("--warmup", default=None, help="short wav to warm up on (discarded)")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    nthreads = torch.get_num_threads()

    device = torch.device("cpu")
    dtype = torch.float32

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, dtype=dtype, device_map="cpu"
    ).eval()
    model.tie_weights()
    assert model.lm_head.weight.device.type != "meta", "tie_weights() failed"
    proc = build_genuine_processor(args.model_dir)
    load_t = time.time() - t0

    def run(wav: str, max_new: int):
        msgs = build_transcription_messages(wav)
        t = time.time()
        with torch.no_grad():
            res = generate_transcription(
                model, proc, msgs, max_new_tokens=max_new,
                do_sample=False, device=device, dtype=dtype,
            )
        return time.time() - t, res["text"], res["generated_tokens"]

    if args.warmup:
        run(args.warmup, 32)

    info = sf.info(args.wav)
    dur = info.frames / info.samplerate
    gen_t, text, ntok = run(args.wav, args.max_new)

    print(f"UPSTREAM torch-cpu threads={nthreads} load={load_t:.1f}s")
    print(f"audio={dur:.1f}s generate={gen_t:.1f}s RTF={gen_t / dur:.3f} tokens={ntok}")
    print("text:", text)


if __name__ == "__main__":
    main()
