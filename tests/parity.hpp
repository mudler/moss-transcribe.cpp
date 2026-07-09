#pragma once
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace mttest {

// Load an f32 tensor (flattened, row-major) by name from a baseline gguf.
inline bool load_baseline(const std::string& path, const std::string& name,
                          std::vector<float>& out, std::vector<int64_t>& shape) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        std::fprintf(stderr, "[parity] tensor '%s' not found in %s\n", name.c_str(), path.c_str());
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    shape.clear();
    // Report shape outer..inner (slowest to fastest varying dimension)
    for (int i = ggml_n_dims(t) - 1; i >= 0; --i) {
        shape.push_back(t->ne[i]);
    }
    size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    std::memcpy(out.data(), t->data, n * sizeof(float));
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

// Compare got vs ref; returns true if the tensors match within tolerance.
// Reports max/mean abs diff AND cosine similarity to stderr.
// Passes if: max_abs <= atol
//         OR (cos >= 0.99999 AND max_abs <= rtol * std(ref)).
inline bool compare(const std::vector<float>& got, const std::vector<float>& ref,
                    const char* label, float atol, float rtol) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu\n",
                     label, got.size(), ref.size());
        return false;
    }
    if (got.empty()) {
        std::fprintf(stderr, "[%s] n=0 (both empty) -> OK\n", label);
        return true;
    }
    double maxabs = 0.0;
    double sumabs = 0.0;
    size_t worst = 0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    double refsum = 0.0, refsq = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double a = (double)got[i];
        double b = (double)ref[i];
        double d = std::fabs(a - b);
        sumabs += d;
        if (d > maxabs) { maxabs = d; worst = i; }
        dot += a * b;
        na  += a * a;
        nb  += b * b;
        refsum += b;
        refsq  += b * b;
    }
    size_t n = got.size();
    double mean = sumabs / (double)n;
    double cos  = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    double refmean = refsum / (double)n;
    double refvar  = refsq / (double)n - refmean * refmean;
    if (refvar < 0.0) refvar = 0.0;
    double refstd  = std::sqrt(refvar);

    bool ok = (maxabs <= (double)atol) ||
              (cos >= 0.99999 && maxabs <= (double)rtol * refstd);

    std::fprintf(stderr,
        "[%s] n=%zu max|d|=%.3e mean|d|=%.3e cos=%.8f std(ref)=%.3e "
        "(worst@%zu got=%.5f ref=%.5f) -> %s\n",
        label, n, maxabs, mean, cos, refstd, worst,
        got[worst], ref[worst], ok ? "OK" : "FAIL");
    return ok;
}

// Load an int32 tensor (flattened) by name from a baseline gguf.
inline bool load_baseline_i32(const std::string& path, const std::string& name,
                               std::vector<int32_t>& out) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        std::fprintf(stderr, "[parity] tensor '%s' not found in %s\n", name.c_str(), path.c_str());
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    std::memcpy(out.data(), t->data, n * sizeof(int32_t));
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

// Read a uint32 KV entry from a baseline gguf (0 if absent / unopenable).
inline uint32_t mttest_read_u32(const std::string& path, const std::string& key) {
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) return 0;
    int64_t id = gguf_find_key(g, key.c_str());
    uint32_t v = (id < 0) ? 0u : gguf_get_val_u32(g, id);
    gguf_free(g);
    return v;
}

// Load a string KV entry from a baseline gguf.
inline bool load_kv_str(const std::string& path, const std::string& key,
                         std::string& out) {
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) {
        std::fprintf(stderr, "[parity] key '%s' not found in %s\n", key.c_str(), path.c_str());
        gguf_free(g);
        return false;
    }
    out = std::string(gguf_get_val_str(g, id));
    gguf_free(g);
    return true;
}

} // namespace mttest
