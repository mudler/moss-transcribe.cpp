#include "backend.hpp"

#include "common.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace mt {

namespace {

ggml_backend_t  g_backend  = nullptr;
ggml_gallocr_t  g_gallocr  = nullptr;
std::string     g_name;
std::once_flag  g_once;

std::string lower(const std::string& s) {
    std::string r = s;
    for (auto& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

// Map a user-facing backend name to a ggml device type. Returns true if
// the name was recognized (even when no matching device is registered).
bool name_to_type(const std::string& name_lower, enum ggml_backend_dev_type* out) {
    if (name_lower == "cpu")      { *out = GGML_BACKEND_DEVICE_TYPE_CPU;  return true; }
    if (name_lower == "cuda")     { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    if (name_lower == "metal")    { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    if (name_lower == "vulkan")   { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    if (name_lower == "hipblas")  { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    if (name_lower == "rocm")     { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    if (name_lower == "gpu")      { *out = GGML_BACKEND_DEVICE_TYPE_GPU;  return true; }
    return false;
}

// Try to init a backend whose ggml_backend_dev_get_name matches the
// requested name (case-insensitive substring). Returns null if no
// matching device is loaded.
ggml_backend_t init_named(const std::string& name_lower) {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const char* dev_name = ggml_backend_dev_name(dev);
        if (!dev_name) continue;
        std::string ln = lower(dev_name);
        if (ln.find(name_lower) != std::string::npos) {
            ggml_backend_t b = ggml_backend_dev_init(dev, /*params=*/nullptr);
            if (b) return b;
        }
    }
    return nullptr;
}

void init() {
    // Load any backend shared libraries (libggml-cuda.so etc.) that ggml
    // ships with. No-op for backends compiled in directly (CPU).
    ggml_backend_load_all();

    const char* env = std::getenv("MTD_DEVICE");
    std::string want = env ? lower(env) : "";

    // 1. Honor MTD_DEVICE verbatim if it matches a registered
    // device.
    if (!want.empty()) {
        if (want == "cpu") {
            g_backend = ggml_backend_dev_init(
                ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        } else {
            g_backend = init_named(want);
            if (!g_backend) {
                enum ggml_backend_dev_type t;
                if (name_to_type(want, &t)) {
                    ggml_backend_dev_t dev = ggml_backend_dev_by_type(t);
                    if (dev) g_backend = ggml_backend_dev_init(dev, nullptr);
                }
            }
            if (!g_backend) {
                MT_LOGW("backend: MTD_DEVICE=%s requested but no "
                            "matching device registered — falling back",
                            env);
            }
        }
    }

    // 2. Default: best-available, with GPU preferred over CPU.
    if (!g_backend) {
        g_backend = ggml_backend_init_best();
    }
    // 3. Belt-and-braces CPU fallback.
    if (!g_backend) {
        g_backend = ggml_backend_dev_init(
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    }

    if (!g_backend) {
        // ggml_backend_init_best already falls back to CPU if no GPU; if
        // we still don't have a backend, init_by_type(CPU) at least.
        g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    if (g_backend) {
        const char* name = ggml_backend_name(g_backend);
        g_name = name ? name : "(unnamed)";
        // Use all available cores by default (ggml's "0" actually means the
        // GGML_DEFAULT_N_THREADS=4 fallback, not all cores). Override with the
        // MTD_THREADS env var (useful for matched-thread benchmarking).
        if (ggml_backend_is_cpu(g_backend)) {
            int nt = 0;
            if (const char* e = std::getenv("MTD_THREADS")) nt = std::atoi(e);
            if (nt <= 0) nt = (int) std::thread::hardware_concurrency();
            if (nt <= 0) nt = 4;
            ggml_backend_cpu_set_n_threads(g_backend, nt);
            MT_LOGI("CPU threads: %d", nt);
        }
        // Allocator for graph intermediates. ggml_gallocr_new takes a
        // single buffer type; we use the backend's default.
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
        g_gallocr = ggml_gallocr_new(buft);
        MT_LOGI("backend: %s", g_name.c_str());
    } else {
        MT_LOGE("backend: failed to initialize any backend");
        g_name = "none";
    }
}

}  // namespace

ggml_backend_t backend() {
    std::call_once(g_once, init);
    return g_backend;
}

const char* backend_name() {
    std::call_once(g_once, init);
    return g_name.c_str();
}

bool compute_graph(ggml_cgraph* graph) {
    ggml_backend_t b = backend();
    if (!b || !graph || !g_gallocr) return false;
    if (!ggml_gallocr_alloc_graph(g_gallocr, graph)) {
        MT_LOGE("backend: gallocr_alloc_graph failed");
        return false;
    }
    return ggml_backend_graph_compute(b, graph) == GGML_STATUS_SUCCESS;
}

bool compute_graph_with_inputs(ggml_cgraph* graph,
                               const std::function<void()>& set_inputs) {
    ggml_backend_t b = backend();
    if (!b || !graph || !g_gallocr) return false;
    // Allocate first: input + intermediate tensors get their backend buffers.
    if (!ggml_gallocr_alloc_graph(g_gallocr, graph)) {
        MT_LOGE("backend: gallocr_alloc_graph failed");
        return false;
    }
    // Now the input leaves have real (possibly device) buffers — upload data.
    if (set_inputs) set_inputs();
    return ggml_backend_graph_compute(b, graph) == GGML_STATUS_SUCCESS;
}

ggml_backend_buffer_t allocate_ctx_tensors(ggml_context* ctx) {
    ggml_backend_t b = backend();
    if (!b || !ctx) return nullptr;
    return ggml_backend_alloc_ctx_tensors(ctx, b);
}

bool backend_supports_flash_attn() {
    static std::once_flag detect_once;
    static bool supported = false;
    std::call_once(detect_once, [] {
        const char* env = std::getenv("MTD_FLASH_ATTN");
        if (env && (std::string(env) == "0" || lower(env) == "false" || lower(env) == "off")) {
            MT_LOGI("backend: flash-attn disabled via MTD_FLASH_ATTN=%s", env);
            supported = false;
            return;
        }

        ggml_backend_t b = backend();
        if (!b) {
            supported = false;
            return;
        }

        // Build a tiny dummy FA op (no allocation) and ask the backend
        // whether it can run it. Shapes are arbitrary - we only need
        // the op kind + dtypes to match what we'd build for real.
        struct ggml_init_params ip {};
        ip.mem_size  = ggml_tensor_overhead() * 16;
        ip.no_alloc  = true;
        ggml_context* ctx = ggml_init(ip);
        const int hd = 128, n_h = 8, n_kv = 2, seq = 16;
        ggml_tensor* q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, n_h,  1);
        ggml_tensor* k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, n_kv, 1);
        ggml_tensor* v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, n_kv, 1);
        ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, seq, seq);
        ggml_tensor* op   = ggml_flash_attn_ext(ctx, q, k, v, mask,
                                                1.0f / 11.31f, 0.0f, 0.0f);
        supported = op && ggml_backend_supports_op(b, op);
        ggml_free(ctx);

        MT_LOGI("backend: flash-attn %s on %s",
                    supported ? "available" : "unavailable",
                    g_name.c_str());
    });
    return supported;
}

bool read_tensor_f32(const struct ggml_tensor* t, std::vector<float>* out) {
    if (!t || !out || t->type != GGML_TYPE_F32) return false;
    const int64_t n = ggml_nelements(t);
    out->resize((size_t)n);
    // ggml_backend_tensor_get takes a const tensor and issues one DtoH copy
    // (a plain memcpy on CPU backends).
    ggml_backend_tensor_get(t, out->data(), 0, (size_t)n * sizeof(float));
    return true;
}

bool read_tensor_i32(const struct ggml_tensor* t, std::vector<int32_t>* out) {
    if (!t || !out || t->type != GGML_TYPE_I32) return false;
    const int64_t n = ggml_nelements(t);
    out->resize((size_t)n);
    ggml_backend_tensor_get(t, out->data(), 0, (size_t)n * sizeof(int32_t));
    return true;
}

}  // namespace mt
