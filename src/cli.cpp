#include "moss_transcribe.h"
#include "model_loader.hpp"
#include "transcribe.hpp"
#include "subtitle.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
            "usage: moss-transcribe transcribe <model.gguf> <audio.wav> "
            "[--max-new N] [--format text|srt|ass|json]\n");
        return 2;
    }
    const char* gguf = argv[2];
    const char* wav  = argv[3];
    int max_new = -1;  // resolved from config below if not overridden
    std::string format = "text";
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
            max_new = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            format = argv[++i];
        }
    }
    if (format != "text" && format != "srt" && format != "ass" && format != "json") {
        std::fprintf(stderr, "unknown --format '%s' (want text|srt|ass|json)\n",
                     format.c_str());
        return 2;
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

    if (format == "text") {
        // Byte-identical to the historical raw output.
        std::printf("%s\n", text.c_str());
    } else if (format == "json") {
        // Raw parsed segments (no timing normalization).
        auto segs = mt::subtitle_segments_from_transcript(text, /*postprocess=*/false);
        std::string out = mt::to_json(segs);
        std::fwrite(out.data(), 1, out.size(), stdout);
    } else {
        // srt / ass: normalize (merge/split/overlap) then export.
        auto segs = mt::subtitle_segments_from_transcript(text, /*postprocess=*/true);
        std::string out = (format == "srt") ? mt::to_srt(segs)
                                            : mt::to_ass(segs);
        std::fwrite(out.data(), 1, out.size(), stdout);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// quantize: re-quantize the SAME allowlisted linear weights as the converter's
// f16/q8_0 path to a target ggml type -- including the K-quants (Q4_K/Q5_K/Q6_K)
// that the Python gguf writer can't produce. Every non-allowlisted tensor (and
// any allowlisted tensor that isn't shape-eligible) is copied verbatim in its
// stored type. All KV metadata is copied unchanged.
//
// Ported from parakeet.cpp's cmd_quantize; the only change is the allowlist,
// which mirrors the converter's _QUANTIZABLE_PATTERNS (token_embd + qwen3
// attn/ffn + whisper-enc attn/ffn + adaptor fc).
// ---------------------------------------------------------------------------

static bool is_quantizable_name(const std::string& n) {
    if (n == "token_embd.weight") return true;
    if (n == "adaptor.fc1.w" || n == "adaptor.fc2.w") return true;
    // qwen3.blk.<d>.<rest> / enc.blk.<d>.<rest>
    auto blk_rest = [](const std::string& s, const char* prefix, std::string& rest) -> bool {
        size_t p = std::strlen(prefix);
        if (s.rfind(prefix, 0) != 0) return false;
        size_t i = p, start = p;
        while (i < s.size() && std::isdigit((unsigned char)s[i])) ++i;
        if (i == start || i >= s.size() || s[i] != '.') return false;
        rest = s.substr(i + 1);
        return true;
    };
    std::string rest;
    if (blk_rest(n, "qwen3.blk.", rest)) {
        return rest == "attn_q.weight" || rest == "attn_k.weight" ||
               rest == "attn_v.weight" || rest == "attn_o.weight" ||
               rest == "ffn_gate.weight" || rest == "ffn_up.weight" || rest == "ffn_down.weight";
    }
    if (blk_rest(n, "enc.blk.", rest)) {
        return rest == "attn_q.w" || rest == "attn_k.w" ||
               rest == "attn_v.w" || rest == "attn_out.w" ||
               rest == "ffn_1.w" || rest == "ffn_2.w";
    }
    return false;
}

static bool parse_quant_type(const std::string& s, ggml_type& out) {
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (t == "q4_0") { out = GGML_TYPE_Q4_0; return true; }
    if (t == "q5_0") { out = GGML_TYPE_Q5_0; return true; }
    if (t == "q8_0") { out = GGML_TYPE_Q8_0; return true; }
    if (t == "q4_k") { out = GGML_TYPE_Q4_K; return true; }
    if (t == "q5_k") { out = GGML_TYPE_Q5_K; return true; }
    if (t == "q6_k") { out = GGML_TYPE_Q6_K; return true; }
    return false;
}

static bool is_k_quant(ggml_type t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K;
}

static int cmd_quantize(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: moss-transcribe quantize <in.gguf> <out.gguf> "
            "<q4_0|q5_0|q8_0|q4_k|q5_k|q6_k>\n");
        return 2;
    }
    const std::string in_path  = argv[0];
    const std::string out_path = argv[1];
    ggml_type qtype;
    if (!parse_quant_type(argv[2], qtype)) {
        std::fprintf(stderr,
            "moss-transcribe quantize: unknown type '%s' "
            "(want q4_0|q5_0|q8_0|q4_k|q5_k|q6_k)\n", argv[2]);
        return 2;
    }

    // Load the source GGUF with a backing ggml_context so tensor data is read in.
    struct ggml_context* src_ctx = nullptr;
    struct gguf_init_params p{ /*no_alloc*/ false, /*ctx*/ &src_ctx };
    struct gguf_context* src = gguf_init_from_file(in_path.c_str(), p);
    if (!src) {
        std::fprintf(stderr, "failed to open %s\n", in_path.c_str());
        return 1;
    }

    const int64_t block = ggml_blck_size(qtype);  // 32 for q*_0, 256 for K-quants

    // Destination: empty gguf, copy all KV verbatim, then add tensors.
    struct gguf_context* dst = gguf_init_empty();
    gguf_set_kv(dst, src);  // copies every KV pair unchanged

    const int64_t nt = gguf_get_n_tensors(src);
    // Holds quantized buffers alive until gguf_write_to_file copies them out.
    std::vector<std::vector<uint8_t>> quant_bufs;
    quant_bufs.reserve(static_cast<size_t>(nt));
    struct ggml_init_params meta_p{ /*mem_size*/ ggml_tensor_overhead() * (nt + 1),
                                    /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    struct ggml_context* meta = ggml_init(meta_p);

    int n_quant = 0, n_kept = 0, n_skipped = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char* name = gguf_get_tensor_name(src, i);
        struct ggml_tensor* t = ggml_get_tensor(src_ctx, name);

        bool do_quant = false;
        if (is_quantizable_name(name) && t->type == GGML_TYPE_F32) {
            const bool two_d  = (t->ne[2] == 1 && t->ne[3] == 1 &&
                                 t->ne[0] >= 32 && t->ne[1] >= 32);
            const bool blk_ok = (t->ne[0] % block == 0);
            if (two_d && blk_ok) {
                do_quant = true;
            } else if (two_d && !blk_ok) {
                std::fprintf(stderr,
                    "  keep F32: %-48s ne0=%lld not divisible by %s block %lld\n",
                    name, (long long)t->ne[0],
                    is_k_quant(qtype) ? "K-quant superblock" : "block",
                    (long long)block);
                ++n_skipped;
            }
        }

        if (do_quant) {
            const int64_t n_per_row = t->ne[0];
            const int64_t nrows     = t->ne[1];
            const size_t  out_bytes = ggml_row_size(qtype, n_per_row) * (size_t)nrows;
            quant_bufs.emplace_back(out_bytes);
            const float* fsrc = (const float*)t->data;
            // imatrix = NULL: none of q4_0/q5_0/q8_0/q4_k/q5_k/q6_k require one.
            ggml_quantize_chunk(qtype, fsrc, quant_bufs.back().data(),
                                /*start*/ 0, nrows, n_per_row, /*imatrix*/ nullptr);
            struct ggml_tensor* q = ggml_new_tensor_2d(meta, qtype, n_per_row, nrows);
            ggml_set_name(q, name);
            q->data = quant_bufs.back().data();
            gguf_add_tensor(dst, q);
            ++n_quant;
        } else {
            gguf_add_tensor(dst, t);
            ++n_kept;
        }
    }

    const bool ok = gguf_write_to_file(dst, out_path.c_str(), /*only_meta*/ false);

    std::printf("quantize: %s -> %s [%s]\n", in_path.c_str(), out_path.c_str(), argv[2]);
    std::printf("  quantized %d tensor(s), copied %d verbatim, %d allowlisted kept F32 (block).\n",
                n_quant, n_kept, n_skipped);

    ggml_quantize_free();
    ggml_free(meta);
    gguf_free(dst);
    gguf_free(src);
    ggml_free(src_ctx);

    if (!ok) {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: moss-transcribe <subcommand>\n"); return 2; }
    if (std::strcmp(argv[1], "version") == 0) { std::printf("%s\n", mt::version()); return 0; }
    if (std::strcmp(argv[1], "info") == 0) { return cmd_info(argc, argv); }
    if (std::strcmp(argv[1], "transcribe") == 0) { return cmd_transcribe(argc, argv); }
    if (std::strcmp(argv[1], "quantize") == 0) { return cmd_quantize(argc - 2, argv + 2); }
    std::fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    return 2;
}
