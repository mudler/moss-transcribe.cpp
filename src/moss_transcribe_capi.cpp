#include "moss_transcribe_capi.h"

#include "model_loader.hpp"
#include "transcribe.hpp"
#include "audio_io.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

// ABI version. Bump on any breaking change to the C-API surface.
#define MOSS_TRANSCRIBE_CAPI_ABI_VERSION 1

// The opaque context: a loaded (and f16->f32 promoted) model plus a buffer for
// the last error message. The model is loaded once and reused across calls.
struct moss_transcribe_ctx {
    std::unique_ptr<mt::ModelLoader> loader;
    std::string last_error;
};

namespace {

// malloc a NUL-terminated copy of `s` so a C consumer frees it with free()
// (matching moss_transcribe_capi_free_string). Returns NULL on OOM.
char* dup_to_c(const std::string& s) {
    char* buf = static_cast<char*>(std::malloc(s.size() + 1));
    if (!buf) return nullptr;
    std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = '\0';
    return buf;
}

// Resolve the effective max_new-tokens budget: `max_new <= 0` means use the
// GGUF's default_max_new_tokens (falling back to 5120 if the model set none),
// matching the CLI.
int resolve_max_new(const mt::ModelLoader& m, int max_new) {
    if (max_new > 0) return max_new;
    const int def = m.config().default_max_new_tokens;
    return def > 0 ? def : 5120;
}

} // namespace

extern "C" int moss_transcribe_capi_abi_version(void) {
    return MOSS_TRANSCRIBE_CAPI_ABI_VERSION;
}

extern "C" moss_transcribe_ctx* moss_transcribe_capi_load(const char* gguf_path) {
    if (!gguf_path) return nullptr;
    try {
        auto loader = std::make_unique<mt::ModelLoader>();
        if (!loader->load(gguf_path)) return nullptr;  // bad/missing GGUF
        // ggml's CPU element-wise ops don't auto-cast f32 + f16, so promote the
        // small f16 tensors up front (same as the CLI transcribe path).
        loader->promote_small_f16_to_f32();
        auto* ctx = new (std::nothrow) moss_transcribe_ctx();
        if (!ctx) return nullptr;
        ctx->loader = std::move(loader);
        return ctx;
    } catch (...) {
        // Never let an exception cross the boundary.
        return nullptr;
    }
}

extern "C" void moss_transcribe_capi_free(moss_transcribe_ctx* ctx) {
    delete ctx;  // safe on nullptr; ~unique_ptr releases the model.
}

extern "C" char* moss_transcribe_capi_transcribe_path(moss_transcribe_ctx* ctx,
                                                      const char* wav_path,
                                                      int max_new) {
    if (!ctx) return nullptr;
    if (!ctx->loader) { ctx->last_error = "context has no loaded model"; return nullptr; }
    if (!wav_path)    { ctx->last_error = "wav_path is NULL"; return nullptr; }
    try {
        const int budget = resolve_max_new(*ctx->loader, max_new);
        std::string text = mt::transcribe_wav(*ctx->loader, wav_path, budget);
        if (text.empty()) { ctx->last_error = "transcription failed"; return nullptr; }
        ctx->last_error.clear();
        char* out = dup_to_c(text);
        if (!out) { ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" char* moss_transcribe_capi_transcribe_pcm(moss_transcribe_ctx* ctx,
                                                     const float* samples,
                                                     int n_samples, int sample_rate,
                                                     int max_new) {
    if (!ctx) return nullptr;
    if (!ctx->loader) { ctx->last_error = "context has no loaded model"; return nullptr; }
    if (!samples || n_samples < 0) { ctx->last_error = "invalid samples buffer"; return nullptr; }
    if (sample_rate <= 0) { ctx->last_error = "invalid sample rate"; return nullptr; }
    try {
        std::vector<float> pcm(samples, samples + n_samples);
        // Resample to 16 kHz mono if needed (input is assumed mono).
        if (sample_rate != 16000) {
            pcm = mt::resample_linear(pcm, sample_rate, 16000);
        }
        const int budget = resolve_max_new(*ctx->loader, max_new);
        std::string text = mt::transcribe_pcm16k(*ctx->loader, pcm, budget);
        if (text.empty()) { ctx->last_error = "transcription failed"; return nullptr; }
        ctx->last_error.clear();
        char* out = dup_to_c(text);
        if (!out) { ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" void moss_transcribe_capi_free_string(char* s) {
    std::free(s);
}

extern "C" const char* moss_transcribe_capi_last_error(moss_transcribe_ctx* ctx) {
    if (!ctx) return "";
    return ctx->last_error.c_str();
}
