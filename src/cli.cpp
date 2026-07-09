#include "moss_transcribe.h"
#include "model_loader.hpp"
#include "transcribe.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int cmd_info(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: moss-transcribe info <model.gguf>\n");
        return 2;
    }
    mt::ModelLoader m;
    if (!m.load(argv[2])) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }
    const auto& c = m.config();
    std::printf("arch=%s text.hidden=%d text.layers=%d audio.d_model=%d audio.layers=%d\n",
                c.arch.c_str(), c.text_hidden, c.text_layers, c.audio_d_model, c.audio_layers);
    std::printf("audio_token_id=%d merge=%d adaptor_in=%d mel_bins=%d n_fft=%d hop=%d\n",
                c.audio_token_id, c.audio_merge_size, c.adaptor_input_dim,
                c.feat_size, c.feat_n_fft, c.feat_hop);
    return 0;
}

static int cmd_transcribe(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: moss-transcribe transcribe <model.gguf> <audio.wav> [--max-new N]\n");
        return 2;
    }
    const char* gguf = argv[2];
    const char* wav  = argv[3];
    int max_new = -1;  // resolved from config below if not overridden
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
            max_new = std::atoi(argv[++i]);
        }
    }
    mt::ModelLoader m;
    if (!m.load(gguf)) { std::fprintf(stderr, "load failed\n"); return 1; }
    m.promote_small_f16_to_f32();
    if (max_new <= 0) {
        max_new = m.config().default_max_new_tokens > 0
                      ? m.config().default_max_new_tokens : 5120;
    }
    std::string text = mt::transcribe_wav(m, wav, max_new);
    if (text.empty()) { std::fprintf(stderr, "transcription failed\n"); return 1; }
    std::printf("%s\n", text.c_str());
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: moss-transcribe <subcommand>\n"); return 2; }
    if (std::strcmp(argv[1], "version") == 0) { std::printf("%s\n", mt::version()); return 0; }
    if (std::strcmp(argv[1], "info") == 0) { return cmd_info(argc, argv); }
    if (std::strcmp(argv[1], "transcribe") == 0) { return cmd_transcribe(argc, argv); }
    std::fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    return 2;
}
