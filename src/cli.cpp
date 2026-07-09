#include "moss_transcribe.h"
#include "model_loader.hpp"

#include <cstdio>
#include <cstring>

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

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: moss-transcribe <subcommand>\n"); return 2; }
    if (std::strcmp(argv[1], "version") == 0) { std::printf("%s\n", mt::version()); return 0; }
    if (std::strcmp(argv[1], "info") == 0) { return cmd_info(argc, argv); }
    std::fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    return 2;
}
