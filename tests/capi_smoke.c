// Minimal C smoke test for the flat moss-transcribe C-API. Compiled as C (not
// C++) and linked against libmoss-transcribe.so to prove the extern-"C" ABI and
// the dlopen-friendly shared library work end to end.
//
//   cc tests/capi_smoke.c -Iinclude -Lbuild-shared -lmoss-transcribe -o /tmp/capi_smoke
//   LD_LIBRARY_PATH=build-shared /tmp/capi_smoke [model.gguf] [audio.wav]

#include "moss_transcribe_capi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    const char* model = (argc > 1) ? argv[1] : "models/moss-transcribe-f32.gguf";
    const char* wav   = (argc > 2) ? argv[2] : "tests/fixtures/short.wav";

    printf("abi_version = %d\n", moss_transcribe_capi_abi_version());

    moss_transcribe_ctx* ctx = moss_transcribe_capi_load(model);
    if (!ctx) {
        fprintf(stderr, "FAIL: load(%s) returned NULL\n", model);
        return 1;
    }

    char* text = moss_transcribe_capi_transcribe_path(ctx, wav, 300);
    if (!text) {
        fprintf(stderr, "FAIL: transcribe_path: %s\n",
                moss_transcribe_capi_last_error(ctx));
        moss_transcribe_capi_free(ctx);
        return 1;
    }

    printf("transcript: %s\n", text);

    int ok = 1;
    if (!strstr(text, "my fellow Americans")) {
        fprintf(stderr, "FAIL: transcript missing \"my fellow Americans\"\n");
        ok = 0;
    }
    if (!strstr(text, "[S01]")) {
        fprintf(stderr, "FAIL: transcript missing \"[S01]\"\n");
        ok = 0;
    }

    moss_transcribe_capi_free_string(text);
    moss_transcribe_capi_free(ctx);

    if (ok) { printf("OK: smoke test passed\n"); return 0; }
    return 1;
}
