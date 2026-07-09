#ifndef MT_GGML_EXTEND_HPP
#define MT_GGML_EXTEND_HPP

// Minimal ggml helpers for moss-transcribe.cpp.
// Intentionally light for now; later milestones add graph helpers
// (attention, conv, etc) following the pattern of stable-diffusion.cpp.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <memory>
#include <string>

namespace mt {

struct GgmlCtxDeleter {
    void operator()(struct ggml_context* c) const noexcept {
        if (c) ggml_free(c);
    }
};
using GgmlCtxPtr = std::unique_ptr<struct ggml_context, GgmlCtxDeleter>;

inline GgmlCtxPtr make_ctx(size_t mem_size, bool no_alloc = false) {
    struct ggml_init_params p = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ no_alloc,
    };
    return GgmlCtxPtr(ggml_init(p));
}

// Like make_ctx, but uses a CALLER-OWNED, reusable mem_buffer instead of having
// ggml malloc one internally. The caller owns `mem_buffer` (keep it alive while
// the ctx is used); ggml_free (via GgmlCtxPtr) frees only the context struct,
// not the buffer. Use for per-call scratch contexts that would otherwise malloc
// mem_size bytes every call.
inline GgmlCtxPtr make_ctx_buf(void* mem_buffer, size_t mem_size, bool no_alloc = false) {
    struct ggml_init_params p = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ mem_buffer,
        /*.no_alloc   =*/ no_alloc,
    };
    return GgmlCtxPtr(ggml_init(p));
}

// y = x @ W^T  (+ b). W is a ggml weight with ne[0]=in, ne[1]=out (matches a
// torch Linear.weight stored (out,in) and loaded row-major). b may be null.
inline struct ggml_tensor* linear(struct ggml_context* ctx, struct ggml_tensor* W,
                                  struct ggml_tensor* b, struct ggml_tensor* x) {
    struct ggml_tensor* y = ggml_mul_mat(ctx, W, x);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}
// LayerNorm with weight+bias over ne[0]. eps typically 1e-5.
inline struct ggml_tensor* layer_norm(struct ggml_context* ctx, struct ggml_tensor* x,
                                      struct ggml_tensor* w, struct ggml_tensor* b, float eps) {
    struct ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    y = ggml_add(ctx, y, b);
    return y;
}

}  // namespace mt

#endif  // MT_GGML_EXTEND_HPP
