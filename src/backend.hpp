#ifndef MT_BACKEND_HPP
#define MT_BACKEND_HPP

// Process-wide ggml backend selection + per-graph compute helper.
//
// This module uses the ggml backend API so the same graphs can run on
// CUDA / Metal / Vulkan / hipBLAS when those are built in (via
// MT_HAVE_CUDA / _METAL / _VULKAN / _HIPBLAS at compile time and
// dlopen-loaded at runtime via ggml_backend_load_all).
//
// Selection order at runtime:
//   1. The backend named by MTD_DEVICE env var (case-insensitive),
//      one of: cuda, metal, vulkan, hipblas, cpu.
//   2. The first GPU-class backend ggml_backend_dev_count reports.
//   3. CPU.
//
// Single global lazy-init: the first call to backend() picks one and
// keeps it for the process lifetime. Pass MTD_DEVICE=cpu to
// force CPU even when GPU backends are available.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <functional>
#include <string>
#include <vector>

namespace mt {

// Returns the singleton backend. Initializes on first call. Never null —
// CPU is the always-available fallback. Lifetime is the process; freed
// at exit.
ggml_backend_t backend();

// Human-readable name of the active backend (e.g. "CUDA", "CPU").
const char* backend_name();

// Compute a graph on the active backend. Allocates intermediate tensors
// on the backend's buffer using a lazily-created `ggml_gallocr_t`, then
// dispatches the compute. Returns true on success.
//
// All graph leaf tensors (inputs + weights) must already live on a
// buffer compatible with the backend (typically allocated via
// allocate_ctx_tensors). For now, weights are CPU-resident and the
// compute ctx is allocated on the active backend; ops that need GPU
// data perform implicit transfers, which is fine for v1 correctness.
bool compute_graph(ggml_cgraph* graph);

// Compute a graph that was built in a `no_alloc=true` context and whose leaf
// INPUT tensors must be filled AFTER the graph allocator runs.
//
// With a no_alloc context an input tensor's backing buffer does not exist
// until `ggml_gallocr_alloc_graph`, so its data cannot be written inline at
// build time (->data is null, and on a GPU backend ->data is device memory
// anyway). This helper sequences the steps correctly:
//   1. ggml_gallocr_alloc_graph(graph)   — input + intermediate buffers exist
//   2. set_inputs()                       — caller uploads input leaf data via
//                                           ggml_backend_tensor_set
//   3. ggml_backend_graph_compute(graph)  — run
// After it returns, outputs are read with ggml_backend_tensor_get.
//
// Inputs that participate must be marked with ggml_set_input so the allocator
// keeps a distinct buffer for them (rather than recycling their memory as a
// reusable intermediate). Returns true on success.
bool compute_graph_with_inputs(ggml_cgraph* graph,
                               const std::function<void()>& set_inputs);

// Allocate all tensors in `ctx` on a buffer compatible with the active
// backend. Use after building a no_alloc ggml_context: allocate, then
// memcpy data into each tensor via ggml_backend_tensor_set.
//
// Returns the allocated buffer, or null on failure. The buffer is owned
// by the caller and must outlive any tensor reads/writes; freed with
// ggml_backend_buffer_free.
ggml_backend_buffer_t allocate_ctx_tensors(ggml_context* ctx);

// Returns true if the active backend can execute ggml_flash_attn_ext
// natively (i.e. without per-op CPU fallback that would force HtoD/DtoH
// copies inside the prefill loop). Result is cached on first call.
//
// Set MTD_FLASH_ATTN=0 to force-disable flash attention even when
// the backend supports it (useful for A/B'ing or when an upstream
// regression breaks our shapes).
bool backend_supports_flash_attn();

// Read an entire tensor's data into a host vector via the backend API
// (device-safe: works whether the tensor lives in host pages or device VRAM).
// `out` is resized to ggml_nelements(t). Returns false on null args or a type
// mismatch (f32 / i32 respectively). One DtoH copy.
bool read_tensor_f32(const struct ggml_tensor* t, std::vector<float>* out);
bool read_tensor_i32(const struct ggml_tensor* t, std::vector<int32_t>* out);

}  // namespace mt

#endif  // MT_BACKEND_HPP
