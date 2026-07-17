// Embed lookup vs reference dequant, for every token_embd type we ship.
//
// Regression test for mudler/LocalAI#10862: on CUDA the token_embd GET_ROWS
// graph GGML_ABORTed for K-quants ("unsupported src0 type: q5_K"), killing the
// whole backend on the first request against a q4_k/q5_k/q6_k GGUF. The fix
// routes unsupported types through a host-side per-row copy + dequant.
//
// The test is model-INDEPENDENT: it builds a synthetic [hidden, vocab]
// token_embd for each type, quantized on the host with ggml_quantize_chunk,
// uploads it to the active backend, then checks embed_rows_f32 against the
// reference to_float dequantization of the same rows. On a CUDA build this
// exercises the host fallback for the K-quants (and would abort without the
// fix); on CPU it exercises the graph path for all types.

#include "backend.hpp"
#include "generate.hpp"
#include "ggml_extend.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

// One K-quant super-block per row keeps every listed type happy
// (QK_K == 256; the classic quants' 32-wide blocks divide it too).
constexpr int kHidden = 256;
constexpr int kVocab  = 64;

// Reference F32 rows for `ids` from the raw (possibly quantized) matrix.
void reference_rows(ggml_type type, const std::vector<uint8_t>& data,
                    const std::vector<int32_t>& ids, std::vector<float>* out) {
    const size_t row_bytes = ggml_row_size(type, kHidden);
    out->resize(ids.size() * (size_t)kHidden);
    for (size_t p = 0; p < ids.size(); ++p) {
        const uint8_t* src = data.data() + (size_t)ids[p] * row_bytes;
        float* dst = out->data() + p * (size_t)kHidden;
        if (type == GGML_TYPE_F32) {
            std::memcpy(dst, src, (size_t)kHidden * sizeof(float));
        } else {
            ggml_get_type_traits(type)->to_float(src, dst, kHidden);
        }
    }
}

int check_type(ggml_type type) {
    // Deterministic random token_embd, then quantize row-wise on the host.
    std::mt19937 rng(1234u + (unsigned)type);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<float> f32((size_t)kHidden * kVocab);
    for (auto& v : f32) v = dist(rng);

    const size_t row_bytes = ggml_row_size(type, kHidden);
    std::vector<uint8_t> data(row_bytes * kVocab);
    if (type == GGML_TYPE_F32) {
        std::memcpy(data.data(), f32.data(), data.size());
    } else {
        const size_t written = ggml_quantize_chunk(
            type, f32.data(), data.data(), 0, kVocab, kHidden,
            /*imatrix=*/nullptr);
        if (written != data.size()) {
            std::fprintf(stderr, "%s: quantize_chunk wrote %zu, want %zu\n",
                         ggml_type_name(type), written, data.size());
            return 1;
        }
    }

    // Upload as a backend tensor, like ModelLoader does with real weights.
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * 4;
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) return 1;
    ggml_tensor* tok = ggml_new_tensor_2d(ctx, type, kHidden, kVocab);
    ggml_backend_buffer_t buffer = mt::allocate_ctx_tensors(ctx);
    if (!buffer) {
        std::fprintf(stderr, "%s: failed to allocate backend tensor\n",
                     ggml_type_name(type));
        ggml_free(ctx);
        return 1;
    }
    ggml_backend_tensor_set(tok, data.data(), 0, data.size());

    // Duplicate + boundary ids on purpose.
    const std::vector<int32_t> ids = {0, 5, 5, kVocab - 1, 17, 0};

    std::vector<float> got;
    const bool ok = mt::embed_rows_f32(tok, ids.data(), (int)ids.size(),
                                       kHidden, &got);
    std::vector<float> want;
    reference_rows(type, data, ids, &want);

    int rc = 0;
    if (!ok || got.size() != want.size()) {
        std::fprintf(stderr, "%s: embed_rows_f32 failed (ok=%d size=%zu/%zu)\n",
                     ggml_type_name(type), (int)ok, got.size(), want.size());
        rc = 1;
    } else {
        float max_abs = 0.f;
        for (size_t i = 0; i < got.size(); ++i) {
            max_abs = std::max(max_abs, std::fabs(got[i] - want[i]));
        }
        // Graph path and host path both use the type's reference dequant;
        // GPU get_rows kernels compute the same scale*q in F32, so any
        // difference beyond rounding noise is a bug.
        const float tol = 1e-6f;
        std::printf("%-6s max_abs_diff=%g (%s)\n", ggml_type_name(type),
                    (double)max_abs, mt::backend_name());
        if (!(max_abs <= tol)) {
            std::fprintf(stderr, "%s: max_abs_diff %g > %g\n",
                         ggml_type_name(type), (double)max_abs, (double)tol);
            rc = 1;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return rc;
}

}  // namespace

int main() {
    const ggml_type types[] = {
        GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    };
    int rc = 0;
    for (ggml_type t : types) rc |= check_type(t);
    return rc;
}
