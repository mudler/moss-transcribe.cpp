#ifndef MOSS_TRANSCRIBE_CAPI_H
#define MOSS_TRANSCRIBE_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

// Flat C-API for moss-transcribe.cpp -- designed for dlopen / cgo / purego
// (LocalAI). All functions are extern "C" and never let a C++ exception cross
// the boundary. The model is loaded ONCE into an opaque `moss_transcribe_ctx`
// and reused across transcribe calls. Returned strings are malloc'd UTF-8 owned
// by the caller and must be released with moss_transcribe_capi_free_string.

// ABI version of this header/implementation. Bump on any breaking change to the
// function signatures or semantics below.
int moss_transcribe_capi_abi_version(void);

// Opaque transcription context (wraps a loaded model + last-error buffer).
typedef struct moss_transcribe_ctx moss_transcribe_ctx;

// Load a GGUF model. Returns an owning context, or NULL on failure.
// The returned context must be released with moss_transcribe_capi_free.
moss_transcribe_ctx* moss_transcribe_capi_load(const char* gguf_path);

// Free a context obtained from moss_transcribe_capi_load. Safe on NULL.
void moss_transcribe_capi_free(moss_transcribe_ctx* ctx);

// Transcribe a WAV file (any sample rate/channel layout; decoded, downmixed to
// mono and resampled to 16 kHz internally). `max_new` caps the number of newly
// generated tokens; `max_new <= 0` uses the GGUF's default_max_new_tokens.
// On success returns a malloc'd, NUL-terminated UTF-8 transcript of the form
// "[start][Sxx]text[end]..." (free with moss_transcribe_capi_free_string). On
// error returns NULL and sets the context's last error (see
// moss_transcribe_capi_last_error).
char* moss_transcribe_capi_transcribe_path(moss_transcribe_ctx* ctx,
                                           const char* wav_path, int max_new);

// Transcribe in-memory mono float PCM (`samples`, length `n_samples`). If
// `sample_rate != 16000` the audio is linearly resampled to 16 kHz first.
// `max_new` is as in moss_transcribe_capi_transcribe_path. On success returns a
// malloc'd UTF-8 transcript (free with moss_transcribe_capi_free_string); on
// error returns NULL and sets the context's last error.
char* moss_transcribe_capi_transcribe_pcm(moss_transcribe_ctx* ctx,
                                          const float* samples, int n_samples,
                                          int sample_rate, int max_new);

// Free a string previously returned by moss_transcribe_capi_transcribe_*.
// Safe on NULL.
void moss_transcribe_capi_free_string(char* s);

// Human-readable description of the last error on `ctx`, or "" if none.
// The returned pointer is owned by the context and valid until the next call on
// it (or until moss_transcribe_capi_free). Returns "" if `ctx` is NULL.
const char* moss_transcribe_capi_last_error(moss_transcribe_ctx* ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MOSS_TRANSCRIBE_CAPI_H
