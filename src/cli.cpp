#include "moss_transcribe.h"
#include <cstdio>
#include <cstring>
int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: moss-transcribe <subcommand>\n"); return 2; }
    if (std::strcmp(argv[1], "version") == 0) { std::printf("%s\n", mt::version()); return 0; }
    std::fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    return 2;
}
