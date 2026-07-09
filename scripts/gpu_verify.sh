#!/usr/bin/env bash
# GPU build + parity + benchmark for moss-transcribe.cpp.
#
# Run on a machine with a GPU and the ggml GPU toolchain installed
# (CUDA: nvcc + cublas/cudart dev; or Vulkan: glslc + headers; or Metal on macOS).
# Verifies that the GPU transcript is IDENTICAL to the CPU/reference transcript
# (bit-exact e2e) and reports the GPU vs CPU real-time factor.
#
#   scripts/gpu_verify.sh <gguf> <wav> [cuda|metal|vulkan|hip]   (default: cuda)
#
# Needs: the repo cloned --recursive (ggml submodule), a GGUF (download one from
# huggingface.co/mudler/moss-transcribe.cpp-gguf), and a 16 kHz mono wav.
set -euo pipefail
GGUF=${1:?usage: gpu_verify.sh <gguf> <wav> [backend]}
WAV=${2:?usage: gpu_verify.sh <gguf> <wav> [backend]}
BE=${3:-cuda}
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"

case "$BE" in
  cuda)   FLAG=-DMT_GGML_CUDA=ON;   BDIR=build-cuda ;;
  metal)  FLAG=-DMT_GGML_METAL=ON;  BDIR=build-metal ;;
  vulkan) FLAG=-DMT_GGML_VULKAN=ON; BDIR=build-vulkan ;;
  hip)    FLAG=-DMT_GGML_HIP=ON;    BDIR=build-hip ;;
  *) echo "unknown backend: $BE"; exit 2 ;;
esac

echo "=== build ($BE) ==="
cmake -B "$BDIR" $FLAG -DMT_BUILD_CLI=ON -DGGML_NATIVE=ON >/dev/null
cmake --build "$BDIR" -j

echo "=== GPU transcribe (device auto-selected: GPU preferred) ==="
GPU_OUT=$("$BDIR/moss-transcribe" transcribe "$GGUF" "$WAV" 2>/tmp/gpu.log) ; echo "$GPU_OUT"
grep -m1 "backend:" /tmp/gpu.log || true

echo "=== CPU transcribe (MTD_DEVICE=cpu) on the same build ==="
CPU_OUT=$(MTD_DEVICE=cpu "$BDIR/moss-transcribe" transcribe "$GGUF" "$WAV" 2>/dev/null)

echo "=== PARITY: GPU transcript == CPU transcript? ==="
if [ "$GPU_OUT" = "$CPU_OUT" ]; then echo "PASS: GPU output byte-identical to CPU"; else
  echo "DIFF:"; diff <(echo "$CPU_OUT") <(echo "$GPU_OUT") || true; fi

echo "=== benchmark: GPU vs CPU wall (warm) ==="
warm() { "$@" >/dev/null 2>&1; }
twall() { local s; s=$( { /usr/bin/time -f "%e" "$@" >/dev/null 2>/tmp/t; } 2>&1; tail -1 /tmp/t ); echo "$s"; }
warm "$BDIR/moss-transcribe" transcribe "$GGUF" "$WAV"
GPU_T=$(twall "$BDIR/moss-transcribe" transcribe "$GGUF" "$WAV")
warm env MTD_DEVICE=cpu "$BDIR/moss-transcribe" transcribe "$GGUF" "$WAV"
CPU_T=$(twall env MTD_DEVICE=cpu "$BDIR/moss-transcribe" transcribe "$GGUF" "$WAV")
DUR=$(python3 -c "import soundfile as sf;i=sf.info('$WAV');print(f'{i.frames/i.samplerate:.1f}')" 2>/dev/null || echo "?")
echo "audio=${DUR}s   GPU wall=${GPU_T}s   CPU wall=${CPU_T}s   (backend=$BE)"

echo "=== optional: full parity suite (needs the baseline fixtures) ==="
if [ -f tests/fixtures/baseline_short.gguf ]; then
  cmake -B "$BDIR" $FLAG -DMT_BUILD_TESTS=ON >/dev/null && cmake --build "$BDIR" -j >/dev/null
  MTD_TEST_GGUF="$GGUF" MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf \
    ctest --test-dir "$BDIR" -L model --output-on-failure || true
else
  echo "(skip: scp tests/fixtures/baseline_short.gguf here to run the per-component cosine gates on GPU)"
fi
