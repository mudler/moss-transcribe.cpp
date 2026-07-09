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
