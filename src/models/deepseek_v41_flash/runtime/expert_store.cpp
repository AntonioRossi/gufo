/* Streamed routed experts for DeepSeek V4.1 through the shared
 * expert_stream::Store. The V4.1 decode keeps its own routes, overlap kernels
 * and ranking; the store owns the cache, reads, prefetch and memory budget. */
#include "expert_store.h"

#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <gufo/ext/experimental/expert_stream.hpp>

#include "model_data_internal.h"

DS41_NAMESPACE_BEGIN

struct ds41_expert_store {
  std::unique_ptr<gufo::expert_stream::Store> store;
  uint64_t requested_limit = 0;
};

namespace {

/* Free host memory left after the cache once the sessions and the prompt
 * cache's backing exist. This is the store's floor and the runner's claim. */
constexpr uint64_t kSessionHeadroom = 2ull << 30;
/* Readable bytes the V4.1 kernels may touch past an expert's last piece. */
constexpr uint32_t kReadMarginBytes = 4096;

bool BuildLayout(const ds41_model* model, const ds41_weights* weights,
                 gufo::expert_stream::Layout* layout) {
  if (!model || !weights || model->n_shards > DS41_MAX_SHARDS)
    return false;
  layout->experts = DS41_N_EXPERT;
  layout->top_k = DS41_N_EXPERT_USED;
  layout->pieces = 3;
  layout->max_batch_tokens = DS41_MAX_PREFILL_BATCH;
  layout->table_experts = DS41_N_EXPERT;
  layout->read_margin_bytes = kReadMarginBytes;
  layout->layers.resize(DS41_N_LAYER);
  for (uint32_t layer = 0; layer < DS41_N_LAYER; ++layer) {
    const auto& w = weights->layer[layer];
    const ds41_tensor* tensors[] = {w.ffn_gate_exps, w.ffn_up_exps,
                                    w.ffn_down_exps};
    auto& pieces = layout->layers[layer];
    pieces.resize(3);
    for (uint32_t p = 0; p < 3; ++p) {
      const ds41_tensor* t = tensors[p];
      if (!t || !t->bytes || t->bytes % DS41_N_EXPERT)
        return false;
      bool found = false;
      for (uint32_t s = 0; s < model->n_shards; ++s) {
        const uint64_t start = model->shard_offset[s];
        if (t->abs_offset >= start &&
            t->abs_offset - start <= model->shard_size[s] &&
            t->bytes <= model->shard_size[s] - (t->abs_offset - start)) {
          pieces[p].file = model->shard_fd[s];
          pieces[p].offset = t->abs_offset - start;
          pieces[p].bytes = t->bytes / DS41_N_EXPERT;
          found = true;
          break;
        }
      }
      if (!found)
        return false;
    }
  }
  return true;
}

gufo::expert_stream::Policy MakePolicy(
    uint64_t cache_limit_bytes,
    const std::vector<gufo::expert_stream::Ranked>& ranking,
    const char* ranking_source) {
  gufo::expert_stream::Policy policy;
  policy.cache_limit_bytes = cache_limit_bytes;
  policy.live_floor_bytes = kSessionHeadroom;
  policy.ranking = ranking;
  policy.ranking_source = ranking_source ? ranking_source : "";
  policy.log_tag = "ds41";
  if (const char* trace = std::getenv("DS41_EXPERT_TRACE"); trace && *trace)
    policy.trace_path = trace;
  return policy;
}

struct OverlapBridge {
  bool (*callback)(void*, const ds41_expert_overlap*);
  void* context;
};

bool OverlapThunk(void* context, const gufo::expert_stream::Overlap* overlap) {
  auto* bridge = static_cast<OverlapBridge*>(context);
  ds41_expert_overlap value;
  value.layer = overlap->layer;
  value.rows = overlap->rows;
  value.miss_rows = overlap->miss_rows;
  value.hit_ids = overlap->hit_ids;
  value.miss_ids = overlap->miss_ids;
  for (uint32_t p = 0; p < 3; ++p)
    value.table[p] = overlap->table ? overlap->table[p] : nullptr;
  return bridge->callback(bridge->context, &value);
}

}  // namespace

ds41_expert_store* ds41_expert_store_create(
    const ds41_model* model, const ds41_weights* weights,
    uint64_t cache_limit_bytes,
    const std::vector<gufo::expert_stream::Ranked>& ranking,
    const char* ranking_source) {
  auto owner =
      std::unique_ptr<ds41_expert_store>(new (std::nothrow) ds41_expert_store);
  if (!owner)
    return nullptr;
  gufo::expert_stream::Layout layout;
  if (!BuildLayout(model, weights, &layout))
    return nullptr;
  const gufo::expert_stream::Policy policy =
      MakePolicy(cache_limit_bytes, ranking, ranking_source);
  owner->store = gufo::expert_stream::Store::Create(layout, policy);
  if (!owner->store)
    return nullptr;
  owner->requested_limit = cache_limit_bytes;
  return owner.release();
}

void ds41_expert_store_destroy(ds41_expert_store* store) { delete store; }

bool ds41_expert_store_attach_tails(ds41_expert_store* store,
                                    const char* path) {
  return store && store->store ? store->store->AttachTails(path) : false;
}

bool ds41_expert_store_fit(ds41_expert_store* store, uint64_t grant_bytes) {
  return store && store->store ? store->store->Fit(grant_bytes) : false;
}

bool ds41_expert_store_prepare(ds41_expert_store* store, uint32_t layer,
                               const int32_t* routes, uint32_t n_tokens,
                               const void** gate, const void** up,
                               const void** down,
                               bool (*overlap)(void*,
                                               const ds41_expert_overlap*),
                               void* overlap_context, hipEvent_t routes_ready) {
  if (!store || !store->store)
    return false;
  const void* tables[3] = {nullptr, nullptr, nullptr};
  OverlapBridge bridge{overlap, overlap_context};
  if (!store->store->Prepare(layer, routes, n_tokens, tables,
                             overlap ? OverlapThunk : nullptr,
                             overlap ? &bridge : nullptr, routes_ready))
    return false;
  *gate = tables[0];
  *up = tables[1];
  *down = tables[2];
  return true;
}

bool ds41_expert_store_prefetch(ds41_expert_store* store, uint32_t layer,
                                const int32_t* experts, uint32_t count,
                                bool wide) {
  return store && store->store
             ? store->store->Prefetch(layer, experts, count, wide)
             : false;
}

bool ds41_expert_store_predict(ds41_expert_store* store, uint32_t layer,
                               hipEvent_t done, const int32_t* ids,
                               uint32_t count, bool wide) {
  return store && store->store
             ? store->store->Predict(layer, done, ids, count, wide)
             : false;
}

void ds41_expert_store_expect(ds41_expert_store* store, uint32_t layer) {
  if (store && store->store)
    store->store->Expect(layer);
}

void ds41_expert_store_predict_flush(ds41_expert_store* store) {
  if (store && store->store)
    store->store->PredictFlush();
}

bool ds41_expert_store_tracing(const ds41_expert_store* store) {
  return store && store->store ? store->store->Tracing() : false;
}

void ds41_expert_store_trace_prediction(ds41_expert_store* store,
                                        uint32_t layer, uint32_t lead,
                                        const int32_t* ids, uint32_t n_tokens,
                                        uint32_t k) {
  if (store && store->store)
    store->store->TracePrediction(layer, lead, ids, n_tokens, k);
}

bool ds41_expert_store_prefetch_idle(const ds41_expert_store* store) {
  return store && store->store ? store->store->PrefetchIdle() : true;
}

ds41_expert_store_stats ds41_expert_store_get_stats(
    const ds41_expert_store* store) {
  ds41_expert_store_stats result;
  if (!store || !store->store)
    return result;
  const auto stats = store->store->GetStats();
  result.uses = stats.uses;
  result.hits = stats.hits;
  result.bytes_read = stats.bytes_read;
  result.tokens = stats.tokens;
  result.stall_ns = stats.stall_ns;
  result.route_wait_ns = stats.route_wait_ns;
  result.read_wait_ns = stats.read_wait_ns;
  result.cache_bytes = stats.cache_bytes;
  result.prefetch_bytes = stats.prefetch_bytes;
  result.prefetch_dropped_bytes = stats.prefetch_dropped_bytes;
  return result;
}

ds41_expert_cache_status ds41_expert_store_get_cache_status(
    const ds41_expert_store* store) {
  ds41_expert_cache_status status;
  if (!store || !store->store)
    return status;
  status.enabled = true;
  status.requested_bytes = store->requested_limit;
  status.allocated_bytes = store->store->GetStats().cache_bytes;
  status.limit_bytes = store->store->LimitBytes();
  return status;
}

void ds41_expert_store_log(const ds41_expert_store* store) {
  if (store && store->store)
    store->store->Log();
}

gufo::expert_stream::Store* ds41_expert_store_native(
    ds41_expert_store* store) {
  return store ? store->store.get() : nullptr;
}

uint64_t ds41_expert_store_headroom(const ds41_expert_store* store) {
  return store && store->store
             ? static_cast<uint64_t>(store->store->FloorBytes())
             : 0;
}

DS41_NAMESPACE_END