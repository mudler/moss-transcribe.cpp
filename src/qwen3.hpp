#ifndef MT_QWEN3_HPP
#define MT_QWEN3_HPP
#include "ggml.h"
#include "model_loader.hpp"
#include <string>
namespace mt {
struct Qwen3Hparams { int hidden=0,n_heads=0,n_kv_heads=0,head_dim=0,intermediate=0,n_layers=0,text_vocab=0; float rope_base=1e6f, rms_eps=1e-6f; bool use_rope = true; };
struct Qwen3Layer {
    struct ggml_tensor *attn_norm=nullptr,*attn_q=nullptr,*attn_k=nullptr,*attn_v=nullptr,*attn_o=nullptr,
                       *q_norm=nullptr,*k_norm=nullptr,*ffn_norm=nullptr,*ffn_gate=nullptr,*ffn_up=nullptr,*ffn_down=nullptr;
};
// Load one transformer layer's weights. The prefix overload reads
// `{prefix}.blk.{i}.*` (e.g. "qwen3" or "local"); the 2-arg overload defaults
// to prefix="qwen3" so all V1 callers keep working unchanged.
bool qwen3_load_layer(const ModelLoader& m, const std::string& prefix, int i, Qwen3Layer* out);  // {prefix}.blk.{i}.*
bool qwen3_load_layer(const ModelLoader& m, int i, Qwen3Layer* out);   // qwen3.blk.{i}.*
// On the additive path k_full/v_full are contiguous (gallocr-safe) graph
// outputs. On the device-cache path (k_cache != null) k_store/v_store are the
// in-graph ggml_cpy nodes that write the new columns into the persistent cache
// (already build_forward_expand'd by the helper); k_full/v_full stay null.
struct Qwen3LayerOut { struct ggml_tensor *y=nullptr,*k_full=nullptr,*v_full=nullptr,
                                           *k_store=nullptr,*v_store=nullptr; };
// x:[hidden,T]; pos:int32[T]; mask:[kv,T] additive (0/-inf) or null (= no bias).
// Additive path: k_past/v_past null on prefill (kv=T); returns k_full/v_full.
// Device-cache path (k_cache != null): store the T new columns into k_cache at
// sequence offset past_seq, read the [0:past_seq+T] prefix by view; the helper
// build_forward_expand's the store nodes into gf (ordering: store before read).
Qwen3LayerOut qwen3_layer_forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* pos,
    struct ggml_tensor* mask, struct ggml_tensor* k_past, struct ggml_tensor* v_past,
    const Qwen3Layer& w, const Qwen3Hparams& hp,
    struct ggml_cgraph* gf = nullptr,
    struct ggml_tensor* k_cache = nullptr, struct ggml_tensor* v_cache = nullptr,
    int past_seq = 0);
}  // namespace mt
#endif
