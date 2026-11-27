/* Opens a DeepSeek V4.1 Flash build and reports its memory plan. With
 * --upload the resident weights are copied to the GPU and the memory peak is
 * reported; otherwise the artifacts are only parsed, validated and bound. */
#include <glob.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gufo/ext/host_budget.hpp>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "probe_status.hpp"
#include "src/models/deepseek_v41_flash/runtime/model.h"
#include "src/models/deepseek_v41_flash/runtime/v41_dspark.h"

#if DS41_VARIANT_CODER
using namespace gufo::models::deepseek_v41_flash::coder;
#else
using namespace gufo::models::deepseek_v41_flash::v384;
#endif

namespace {
using gufo::tests::ds41::ProbeStages;
using gufo::tests::ds41::Stage;

struct ProbeStatus {
  const char* path = nullptr;
  ProbeStages stages;
  bool upload = false;
  bool ordinary = false;
  bool requested_streaming = false;
  std::uint64_t requested_cache_bytes = 0;
  ds41_expert_cache_status expert_cache{};
  bool expert_cache_read = false;
  unsigned tokenized_lines = 0;
  unsigned tokenized_tokens = 0;

  bool Write(int exit_code) const {
    if (!path)
      return true;
    const std::string temporary = std::string(path) + ".tmp";
    FILE* file = std::fopen(temporary.c_str(), "wx");
    if (!file) {
      std::perror("Cannot create probe status");
      return false;
    }
    const bool written =
        std::fprintf(
            file,
            "{\n  \"schema\": 1,\n  \"complete\": true,\n  \"exit_code\": %d,\n"
            "  \"upload\": %s,\n  \"ordinary\": %s,\n"
            "  \"loading\": \"%s\",\n  \"tokenization\": \"%s\",\n"
            "  \"prefill\": \"%s\",\n  \"decode\": \"%s\",\n"
            "  \"reason\": \"%s\",\n"
            "  \"tokenized_lines\": %u,\n  \"tokenized_tokens\": %u,\n"
            "  \"requested_stream_experts\": %s,\n"
            "  \"requested_expert_cache_bytes\": %llu,\n"
            "  \"expert_cache\": {\"enabled\": %s, \"requested_bytes\": %llu, "
            "\"allocated_bytes\": %llu, \"limit_bytes\": %llu}\n}\n",
            exit_code, upload ? "true" : "false", ordinary ? "true" : "false",
            stages.Name(Stage::kLoading), stages.Name(Stage::kTokenization),
            stages.Name(Stage::kPrefill), stages.Name(Stage::kDecode),
            stages.Reason(), tokenized_lines, tokenized_tokens,
            requested_streaming ? "true" : "false",
            static_cast<unsigned long long>(requested_cache_bytes),
            expert_cache.enabled ? "true" : "false",
            static_cast<unsigned long long>(expert_cache.requested_bytes),
            static_cast<unsigned long long>(expert_cache.allocated_bytes),
            static_cast<unsigned long long>(expert_cache.limit_bytes)) >= 0;
    const bool closed = std::fclose(file) == 0;
    if (!written || !closed || std::rename(temporary.c_str(), path) != 0) {
      std::perror("Cannot write probe status");
      std::remove(temporary.c_str());
      return false;
    }
    return true;
  }
};

using DecoderPtr =
    std::unique_ptr<ds41_decoder, decltype(&ds41_decoder_destroy)>;
using DsparkStatePtr =
    std::unique_ptr<ds41_dspark_state, decltype(&ds41_dspark_state_destroy)>;
using SessionPtr = std::unique_ptr<ds41_session, decltype(&ds41_session_free)>;
struct FileCloser {
  void operator()(FILE* file) const {
    if (file)
      std::fclose(file);
  }
};
using FilePtr = std::unique_ptr<FILE, FileCloser>;

/* Reads the cache status once, when the path has no reading yet, then closes
 * the engine. This covers an exception before the fit. */
struct EngineCloser {
  ProbeStatus* status;

  void operator()(ds41_engine* engine) const {
    if (!engine)
      return;
    if (!status->expert_cache_read) {
      status->expert_cache = ds41_engine_expert_cache_status(engine);
      status->expert_cache_read = true;
    }
    ds41_engine_close(engine);
  }
};

/* One fit for every streamed probe path. The store limit stays zero until
 * Fit, so each path fits after it allocates its decoders, sessions, and
 * buffers and before its first evaluation. This helper reports a failure
 * through the prefill transition. It records the fit reading when the caller
 * asks for it, and always on failure; ordinary decode uses its final reading
 * on success. */
int FitStreamedExperts(ds41_engine* engine, ProbeStatus* status, bool record) {
  if (!status->requested_streaming) {
    if (record) {
      status->expert_cache = ds41_engine_expert_cache_status(engine);
      status->expert_cache_read = true;
    }
    return 0;
  }
  const std::uint64_t available = gufo::ext::HostAvailableBytes();
  const std::uint64_t headroom = ds41_engine_expert_cache_headroom(engine);
  const std::uint64_t grant = available > headroom ? available - headroom : 0;
  if (ds41_engine_fit_experts(engine, grant) != 0) {
    status->expert_cache = ds41_engine_expert_cache_status(engine);
    status->expert_cache_read = true;
    return status->stages.Transition(Stage::kPrefill, 0, 1,
                                     "expert_cache_fit_failed");
  }
  if (record) {
    status->expert_cache = ds41_engine_expert_cache_status(engine);
    status->expert_cache_read = true;
  }
  return 0;
}

std::uint64_t MeminfoKiB(const char* key) {
  std::ifstream in("/proc/meminfo");
  std::string name;
  std::uint64_t value = 0;
  std::string unit;
  while (in >> name >> value >> unit)
    if (name == key)
      return value;
  return 0;
}

std::uint64_t StatusKiB(const char* key) {
  std::ifstream in("/proc/self/status");
  std::string line;
  const std::size_t len = std::strlen(key);
  while (std::getline(in, line))
    if (line.compare(0, len, key) == 0)
      return std::stoull(line.substr(len + 1));
  return 0;
}

std::uint64_t GttUsedBytes() {
  glob_t found{};
  std::uint64_t bytes = 0;
  if (::glob("/sys/class/drm/card*/device/mem_info_gtt_used", 0, nullptr,
             &found) == 0) {
    for (std::size_t i = 0; i < found.gl_pathc; ++i) {
      std::ifstream in(found.gl_pathv[i]);
      std::uint64_t value = 0;
      if (in >> value)
        bytes += value;
    }
  }
  ::globfree(&found);
  return bytes;
}

struct Dump {
  std::string dir;
  uint32_t pos = 0;
  FILE* routes = nullptr;
};

void DumpLayer(void* ctx, uint32_t layer, const float* hidden,
               const int32_t* experts) {
  auto* dump = static_cast<Dump*>(ctx);
  const std::string path = dump->dir + "/pos" + std::to_string(dump->pos) +
                           "_layer" + std::to_string(layer) + ".f32";
  if (FILE* f = std::fopen(path.c_str(), "wb")) {
    std::fwrite(hidden, sizeof(float), 4u * 5120u, f);
    std::fclose(f);
  }
  std::fprintf(dump->routes, "%u %u %d %d %d %d %d %d\n", dump->pos, layer,
               experts[0], experts[1], experts[2], experts[3], experts[4],
               experts[5]);
}

void Report(const char* stage) {
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  std::printf(
      "%-10s MemAvailable %.2f GiB, GTT used %.2f GiB, RSS %.2f GiB (peak %.2f "
      "GiB)\n",
      stage, MeminfoKiB("MemAvailable:") * 1024.0 / kGiB, GttUsedBytes() / kGiB,
      StatusKiB("VmRSS:") * 1024.0 / kGiB, StatusKiB("VmHWM:") * 1024.0 / kGiB);
}
int Argmax(const float* row, std::size_t vocab) {
  int best = 0;
  for (std::size_t i = 1; i < vocab; ++i)
    if (row[i] > row[(std::size_t)best])
      best = (int)i;
  return best;
}

/* Prefills `prompt` into an existing decoder in batches. */
int PrefillDecoder(ds41_decoder* d, const std::vector<int>& prompt,
                   std::vector<float>* last) {
  const uint32_t cap = ds41_decoder_batch_capacity(d);
  for (uint32_t at = 0; at < prompt.size();) {
    const uint32_t n = std::min<uint32_t>(cap, (uint32_t)prompt.size() - at);
    const bool final = at + n == prompt.size();
    if (ds41_decoder_eval(d, prompt.data() + at, n, last->data(), final ? 1 : 0,
                          nullptr, nullptr) != 0)
      return 1;
    at += n;
  }
  return 0;
}

/* --joint-check N: N sessions (prefixes of the prompt) run speculative
 * cycles of 1 to 6 rows each, as one joint evaluation and each on its own
 * decoder; every row's logits must match bit for bit. The timings compare
 * the joint evaluation with the sessions' own evaluations. */
int JointProbe(ds41_engine* engine, const std::vector<int>& prompt,
               uint32_t context, int sessions, int cycles, uint32_t budget,
               ProbeStatus* status) {
  const std::size_t vocab = (std::size_t)ds41_engine_vocab_size(engine);
  context = std::max<uint32_t>(context, (uint32_t)prompt.size() + 4096);
  std::vector<std::vector<int>> heads;
  heads.reserve(sessions);
  std::vector<DecoderPtr> joint, solo;
  std::vector<ds41_decoder*> joint_raw;
  joint.reserve(sessions);
  solo.reserve(sessions);
  joint_raw.reserve(sessions);
  for (int s = 0; s < sessions; ++s) {
    const std::size_t len = std::max<std::size_t>(
        1, prompt.size() - std::min(prompt.size() - 1, (std::size_t)5 * s));
    heads.emplace_back(prompt.begin(), prompt.begin() + (std::ptrdiff_t)len);
    joint.emplace_back(ds41_decoder_create(engine, context),
                       &ds41_decoder_destroy);
    solo.emplace_back(ds41_decoder_create(engine, context),
                      &ds41_decoder_destroy);
    if (!joint.back() || !solo.back())
      return 1;
    joint_raw.push_back(joint.back().get());
  }
  std::vector<std::vector<float>> jrows(sessions), srows(sessions);
  for (int s = 0; s < sessions; ++s) {
    jrows[s].resize(vocab * DS41_DECODER_MAX_SPECULATIVE);
    srows[s].resize(vocab * DS41_DECODER_MAX_SPECULATIVE);
  }
  std::vector<std::vector<int>> batch(
      (std::size_t)sessions, std::vector<int>(DS41_DECODER_MAX_SPECULATIVE));
  std::vector<uint32_t> ns((std::size_t)sessions);
  std::vector<const int*> tokens((std::size_t)sessions);
  std::vector<float*> out((std::size_t)sessions);
  std::vector<int> next((std::size_t)sessions);
  for (int s = 0; s < sessions; ++s) {
    tokens[s] = batch[s].data();
    out[s] = jrows[s].data();
  }
  std::vector<float> last(vocab);
  const int fit = FitStreamedExperts(engine, status, true);
  if (fit != 0)
    return fit;
  for (int s = 0; s < sessions; ++s) {
    if (PrefillDecoder(joint[s].get(), heads[s], &last) != 0 ||
        PrefillDecoder(solo[s].get(), heads[s], &last) != 0)
      return 1;
    next[s] = Argmax(last.data(), vocab);
  }
  std::size_t rows = 0, identical = 0, total_rows = 0;
  double joint_ms = 0.0, solo_ms = 0.0;
  for (int c = 0; c < cycles; ++c) {
    uint32_t used = 0;
    for (int s = 0; s < sessions; ++s) {
      const uint32_t room = budget - used - (sessions - s - 1);
      ns[s] = std::min<uint32_t>(1 + (uint32_t)(c + 2 * s) % 6, room);
      used += ns[s];
      batch[s][0] = next[s];
      for (uint32_t i = 1; i < ns[s]; ++i)
        batch[s][i] =
            prompt[(97 * (std::size_t)c + 13 * s + i) % prompt.size()];
    }
    total_rows += used;
    auto t0 = std::chrono::steady_clock::now();
    if (ds41_decoder_eval_joint(joint_raw.data(), tokens.data(), ns.data(),
                                (uint32_t)sessions, out.data())) {
      std::fprintf(stderr, "joint evaluation failed at cycle %d\n", c);
      return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int s = 0; s < sessions; ++s)
      if (ds41_decoder_eval_speculative(solo[s].get(), tokens[s], ns[s],
                                        srows[s].data()))
        return 1;
    auto t2 = std::chrono::steady_clock::now();
    if (c > 0) {  // the first cycle warms up
      joint_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      solo_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    for (int s = 0; s < sessions; ++s) {
      for (uint32_t i = 0; i < ns[s]; ++i, ++rows) {
        const bool same = std::memcmp(jrows[s].data() + i * vocab,
                                      srows[s].data() + i * vocab,
                                      vocab * sizeof(float)) == 0;
        identical += same;
        if (!same)
          std::printf(
              "joint-check: cycle %d (%u rows) session %d row %u "
              "differs\n",
              c, used, s, i);
      }
      const uint32_t keep = 1 + (uint32_t)(c + s) % ns[s];
      if (ds41_decoder_commit(joint[s].get(), keep) ||
          ds41_decoder_commit(solo[s].get(), keep))
        return 1;
      next[s] = Argmax(jrows[s].data() + (keep - 1) * vocab, vocab);
    }
  }
  std::printf(
      "joint-check: %d sessions, %d cycles, %zu rows (mean %.1f per joint "
      "batch), %zu bit-identical to their own evaluations; joint %.1f ms, own "
      "%.1f ms per cycle\n",
      sessions, cycles, rows, (double)total_rows / cycles, identical,
      cycles > 1 ? joint_ms / (cycles - 1) : 0.0,
      cycles > 1 ? solo_ms / (cycles - 1) : 0.0);
  return identical == rows ? 0 : 1;
}

/* --spec-check: speculative batches of --spec-width rows (default 4, at
 * most DS41_DECODER_MAX_SPECULATIVE) keeping 1, 2, ..., width rows in turn.
 * Pass A drafts the greedy continuation, pass B replaces the rows past the kept
 * prefix with other tokens. Exact rollback leaves both passes with
 * bit-identical logits for every kept row, and every kept logit row must also
 * match serial decode exactly.
 * --spec-bench N: times speculative batches of N rows (0: 1 to 8), commit 1. */
int SpeculativeProbe(ds41_engine* engine, const std::vector<int>& prompt,
                     uint32_t context, int cycles, int bench, int width,
                     ProbeStatus* status) {
  const std::size_t vocab = (std::size_t)ds41_engine_vocab_size(engine);
  context = std::max<uint32_t>(context, (uint32_t)prompt.size() + 4096);
  std::vector<float> last(vocab);
  if (bench >= 0) {
    if (bench > DS41_DECODER_MAX_SPECULATIVE)
      return 2;
    DecoderPtr d(ds41_decoder_create(engine, context), &ds41_decoder_destroy);
    if (!d)
      return 1;
    int batch[DS41_DECODER_MAX_SPECULATIVE];
    std::vector<float> rows(vocab * DS41_DECODER_MAX_SPECULATIVE);
    const int fit = FitStreamedExperts(engine, status, true);
    if (fit != 0)
      return fit;
    if (PrefillDecoder(d.get(), prompt, &last) != 0)
      return 1;
    int token = Argmax(last.data(), vocab);
    for (uint32_t n = bench ? (uint32_t)bench : 1;
         n <= (bench ? (uint32_t)bench : 8); ++n) {
      double ms[7];
      for (int rep = 0; rep < 7; ++rep) {
        /* The greedy token, then prompt text: distinct tokens route to as
         * many experts as a real draft. */
        batch[0] = token;
        for (uint32_t i = 1; i < n; ++i)
          batch[i] = prompt[(97 * (std::size_t)rep + i) % prompt.size()];
        const auto t0 = std::chrono::steady_clock::now();
        if (ds41_decoder_eval_speculative(d.get(), batch, n, rows.data()) ||
            ds41_decoder_commit(d.get(), 1))
          return 1;
        ms[rep] = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
        token = Argmax(rows.data(), vocab);
        /* DS41_PROBE_GAP=1 separates repetitions by an idle gap for GPU
         * profiles; it lowers the clocks, so timings run without it. */
        if (std::getenv("DS41_PROBE_GAP"))
          std::this_thread::sleep_for(std::chrono::milliseconds(30));
      }
      std::sort(ms + 1, ms + 7);  // the first repetition warms up
      std::printf("spec-bench n=%u at %u: median %.1f ms min %.1f ms\n", n,
                  ds41_decoder_pos(d.get()), ms[3], ms[1]);
    }
    return 0;
  }
  /* Greedy continuation for pass A's drafts, from single-token steps. */
  std::vector<int> greedy;
  std::vector<std::vector<float>> greedy_logits;
  if (width <= 0 || width > DS41_DECODER_MAX_SPECULATIVE || cycles < 0)
    return 2;
  if (vocab == 0)
    return 1;
  const auto keep_at = [width](int c) -> std::uint32_t {
    return 1 +
           static_cast<std::uint32_t>(c) % static_cast<std::uint32_t>(width);
  };
  std::uint64_t keep_rows = 0;
  for (int c = 0; c < cycles; ++c)
    keep_rows += keep_at(c);
  if (keep_rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) -
                      static_cast<std::uint64_t>(width) ||
      keep_rows > std::numeric_limits<std::size_t>::max() / vocab ||
      vocab > std::numeric_limits<std::size_t>::max() /
                  static_cast<std::uint64_t>(width))
    return 1;
  const int greedy_rows = static_cast<int>(keep_rows) + width;
  std::vector<float> rows(vocab * static_cast<std::size_t>(width));
  std::vector<std::vector<float>> kept(keep_rows, std::vector<float>(vocab));
  greedy_logits.resize(greedy_rows, std::vector<float>(vocab));
  greedy.reserve(static_cast<std::size_t>(greedy_rows) + 1);
  DecoderPtr d(ds41_decoder_create(engine, context), &ds41_decoder_destroy);
  if (!d)
    return 1;
  const int fit = FitStreamedExperts(engine, status, true);
  if (fit != 0)
    return fit;
  if (PrefillDecoder(d.get(), prompt, &last) != 0)
    return 1;
  greedy.push_back(Argmax(last.data(), vocab));
  for (int i = 0; i < greedy_rows; ++i) {
    if (ds41_decoder_step(d.get(), greedy.back(), last.data(), nullptr,
                          nullptr))
      return 1;
    greedy_logits[i] = last;
    greedy.push_back(Argmax(last.data(), vocab));
  }
  d.reset();
  std::size_t identical = 0, agree = 0, as_single = 0;
  double worst = 0.0, worst_single = 0.0;
  for (int pass = 0; pass < 2; ++pass) {
    DecoderPtr pd(ds41_decoder_create(engine, context), &ds41_decoder_destroy);
    if (!pd)
      return 1;
    if (PrefillDecoder(pd.get(), prompt, &last) != 0)
      return 1;
    std::size_t at = 0;  // greedy index of the next row's token
    for (int c = 0; c < cycles; ++c) {
      const uint32_t keep = keep_at(c);
      int batch[DS41_DECODER_MAX_SPECULATIVE];
      for (uint32_t r = 0; r < (uint32_t)width; ++r)
        batch[r] =
            pass == 1 && r >= keep ? (int)(1000 + 37 * r + c) : greedy[at + r];
      if (ds41_decoder_eval_speculative(pd.get(), batch, width, rows.data()) ||
          ds41_decoder_commit(pd.get(), keep))
        return 1;
      for (uint32_t r = 0; r < keep; ++r) {
        const std::size_t k = at + r;
        const float* row = rows.data() + r * vocab;
        if (pass == 0) {
          kept[k].assign(row, row + vocab);
          agree += Argmax(row, vocab) == greedy[k + 1];
        } else {
          double diff = 0.0, diff_single = 0.0;
          for (std::size_t v = 0; v < vocab; ++v) {
            if (!std::isfinite(kept[k][v]) || !std::isfinite(row[v]) ||
                !std::isfinite(greedy_logits[k][v])) {
              std::fprintf(
                  stderr,
                  "non-finite speculative/serial logit at row %zu token %zu\n",
                  k, v);
              return 1;
            }
            diff = std::max(diff, (double)std::fabs(kept[k][v] - row[v]));
            diff_single =
                std::max(diff_single,
                         (double)std::fabs(kept[k][v] - greedy_logits[k][v]));
          }
          identical += diff == 0.0;
          as_single += diff_single == 0.0;
          worst = std::max(worst, diff);
          worst_single = std::max(worst_single, diff_single);
        }
      }
      at += keep;
    }
  }
  std::printf(
      "spec-check: %zu kept rows, %zu bit-identical between passes (max |diff| "
      "%.3g); against single-token decode: argmax matches on %zu, logits "
      "bit-identical on %zu (max |diff| %.3g)\n",
      kept.size(), identical, worst, agree, as_single, worst_single);
  return identical == kept.size() && as_single == kept.size() ? 0 : 1;
}
}  // namespace

/* --dspark-generate N: greedy decode of N tokens after the prompt, first
 * token by token, then in speculative cycles (prompt lookup, plus DSpark
 * drafts with --dspark); the outputs must match.
 * --dspark-sweep repeats the DSpark run with at most 1 to 5 drafts. */
int DsparkProbe(ds41_engine* engine, const std::vector<int>& prompt,
                uint32_t context, int n_tokens, bool sweep,
                ProbeStatus* status) {
  const int ctx =
      (int)std::max<std::size_t>(context, prompt.size() + n_tokens + 16);
  ds41_session* raw = nullptr;
  if (ds41_session_create(&raw, engine, ctx, ds41_engine_has_dspark(engine)) !=
      0) {
    std::fprintf(stderr, "DSpark session creation failed\n");
    return 1;
  }
  SessionPtr session_owner(raw, &ds41_session_free);
  ds41_session* session = session_owner.get();
  std::vector<int> plain;
  std::vector<int> spec;
  plain.reserve((std::size_t)n_tokens);
  spec.reserve((std::size_t)n_tokens);
  const int fit = FitStreamedExperts(engine, status, true);
  if (fit != 0)
    return fit;
  char error[256] = {};
  ds41_tokens tokens{const_cast<int*>(prompt.data()), (int)prompt.size(),
                     (int)prompt.size()};
  const auto now = [] { return std::chrono::steady_clock::now(); };
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  if (ds41_session_sync(session, &tokens, 0, error, sizeof(error)) != 0) {
    std::fprintf(stderr, "sync failed: %s\n", error);
    return 1;
  }
  auto t0 = now();
  while ((int)plain.size() < n_tokens) {
    const int token = ds41_session_argmax(session);
    plain.push_back(token);
    if (ds41_session_eval(session, token, error, sizeof(error)) != 0) {
      std::fprintf(stderr, "eval failed: %s\n", error);
      return 1;
    }
  }
  const double plain_ms = ms(t0, now());
  ds41_session_invalidate(session);
  if (ds41_session_sync(session, &tokens, 0, error, sizeof(error)) != 0) {
    std::fprintf(stderr, "sync failed: %s\n", error);
    return 1;
  }
  const int widths = sweep ? 5 : 1;
  for (int width = 1; width <= widths; width++) {
    if (width > 1) {
      ds41_session_invalidate(session);
      if (ds41_session_sync(session, &tokens, 0, error, sizeof(error)) != 0)
        return 1;
    }
    spec.clear();
    int block[32];
    int produced = 0;
    uint64_t base_drafted = 0, base_accepted = 0, base_steps = 0, base_full = 0;
    ds41_session_dspark_stats(session, nullptr, nullptr, &base_drafted,
                              &base_accepted, nullptr, nullptr, &base_full,
                              &base_steps, nullptr, nullptr);
    t0 = now();
    while ((int)spec.size() < n_tokens) {
      ds41_session_dspark_batch_item item{
          session,
          block,
          std::min(32, n_tokens - (int)spec.size()),
          widths > 1 ? (uint32_t)width : 5u,
          &produced,
          nullptr,
          false};
      if (ds41_sessions_dspark_step_batch(&item, 1, error, sizeof(error)) !=
          0) {
        std::fprintf(stderr, "DSpark step failed: %s\n", error);
        return 1;
      }
      spec.insert(spec.end(), block, block + produced);
    }
    const double spec_ms = ms(t0, now());
    uint64_t drafted = 0, accepted = 0, steps = 0, full = 0;
    ds41_session_dspark_stats(session, nullptr, nullptr, &drafted, &accepted,
                              nullptr, nullptr, &full, &steps, nullptr,
                              nullptr);
    drafted -= base_drafted;
    accepted -= base_accepted;
    steps -= base_steps;
    full -= base_full;
    if (widths > 1)
      std::printf("width %d: ", width);
    int same = 0;
    while (same < n_tokens && plain[same] == spec[same])
      same++;
    std::printf("plain %d tokens %.1f ms (%.2f tok/s)\n", n_tokens, plain_ms,
                n_tokens * 1000.0 / plain_ms);
    std::printf(
        "dspark %zu tokens %.1f ms (%.2f tok/s), %llu cycles, drafted %llu "
        "accepted %llu (%.1f%%), %.2f "
        "tokens/cycle, %llu full blocks\n",
        spec.size(), spec_ms, spec.size() * 1000.0 / spec_ms,
        (unsigned long long)steps, (unsigned long long)drafted,
        (unsigned long long)accepted,
        drafted ? 100.0 * accepted / drafted : 0.0,
        steps ? (double)spec.size() / steps : 0.0, (unsigned long long)full);
    uint64_t lookup_cycles = 0, lookup_copied = 0, lookup_accepted = 0;
    ds41_session_lookup_stats(session, &lookup_cycles, &lookup_copied,
                              &lookup_accepted);
    std::printf("lookup: %llu cycles, copied %llu accepted %llu\n",
                (unsigned long long)lookup_cycles,
                (unsigned long long)lookup_copied,
                (unsigned long long)lookup_accepted);
    std::printf("greedy match: %d of %d tokens%s\n", same, n_tokens,
                same == n_tokens ? " (identical)" : "");
    if (same < n_tokens) {
      std::printf("plain:");
      for (int i = same; i < std::min(n_tokens, same + 8); i++)
        std::printf(" %d", plain[i]);
      std::printf("\ndspark:");
      for (int i = same; i < std::min(n_tokens, same + 8); i++)
        std::printf(" %d", spec[i]);
      std::printf("\n");
    }
    if (same < n_tokens)
      return 1;
  }
  return 0;
}

/* --dspark-dump DIR: prefills the prompt, drafts after its argmax and writes
 * the drafter inputs and outputs for tools/ds41/dspark_reference.py:
 * features.f32 ([rows][3 * dim] target features of the last rows),
 * logits.f32 ([block][vocab] head logits before the Markov bias) and
 * draft.txt (first feature position, rows, anchor, drafts). */
int DsparkDump(ds41_engine* engine, const std::vector<int>& prompt,
               const char* dir, ProbeStatus* status) {
  if (prompt.size() < 4)
    return status->stages.Transition(Stage::kPrefill, 0, 1,
                                     "dspark_prefix_too_short");
  constexpr uint32_t kOthers = 1;
  DecoderPtr d(ds41_decoder_create(engine, (uint32_t)prompt.size() + 16),
               &ds41_decoder_destroy);
  DsparkStatePtr s(d ? ds41_dspark_state_create(engine, d.get()) : nullptr,
                   &ds41_dspark_state_destroy);
  std::vector<float> logits((std::size_t)ds41_engine_vocab_size(engine));
  if (!s)
    return 1;
  const uint32_t cap = ds41_decoder_batch_capacity(d.get());
  std::vector<DecoderPtr> od;
  std::vector<DsparkStatePtr> os;
  od.reserve(kOthers);
  os.reserve(kOthers);
  for (uint32_t o = 0; o < kOthers; ++o) {
    const uint32_t len = (uint32_t)prompt.size() - 3 * (o + 1);
    DecoderPtr dd(ds41_decoder_create(engine, len + 16), &ds41_decoder_destroy);
    DsparkStatePtr ss(dd ? ds41_dspark_state_create(engine, dd.get()) : nullptr,
                      &ds41_dspark_state_destroy);
    if (!ss) {
      std::fprintf(stderr, "batched draft: session %u not created\n", o);
      return 1;
    }
    od.push_back(std::move(dd));
    os.push_back(std::move(ss));
  }
  const uint32_t feature_rows =
      std::min<uint32_t>((uint32_t)prompt.size(), DS41_DSPARK_FEATURE_ROWS);
  std::vector<float> host((std::size_t)feature_rows *
                          DS41_DSPARK_TARGET_LAYERS * 5120);
  std::vector<float> draft_logits((std::size_t)DS41_DSPARK_BLOCK *
                                  logits.size());
  std::vector<float> comparison_logits(draft_logits.size());
  const int fit = FitStreamedExperts(engine, status, true);
  if (fit != 0)
    return fit;
  uint32_t pos0 = 0, rows = 0;
  const float* features = nullptr;
  for (uint32_t at = 0; at < prompt.size();) {
    const uint32_t n = std::min<uint32_t>(cap, (uint32_t)prompt.size() - at);
    if (ds41_decoder_eval(d.get(), prompt.data() + at, n, logits.data(), 1,
                          nullptr, nullptr) != 0)
      return 1;
    features = ds41_decoder_features(d.get(), &pos0, &rows);
    /* The ring holds only the dumped rows, so the oracle sees the same keys. */
    ds41_dspark_state_reset(s.get(), pos0);
    if (!ds41_dspark_inject(s.get(), pos0, rows))
      return 1;
    at += n;
  }
  const int anchor = Argmax(logits.data(), logits.size());
  int32_t drafts[DS41_DSPARK_BLOCK];
  if (!ds41_dspark_draft(s.get(), anchor, drafts, DS41_DSPARK_BLOCK))
    return 1;
  if (hipMemcpy(
          host.data(), features,
          (std::size_t)rows * DS41_DSPARK_TARGET_LAYERS * 5120 * sizeof(float),
          hipMemcpyDeviceToHost) != hipSuccess ||
      !ds41_dspark_copy_logits(engine, draft_logits.data()))
    return 1;
  const std::string base(dir);
  std::ofstream features_out(base + "/features.f32", std::ios::binary);
  std::ofstream logits_out(base + "/logits.f32", std::ios::binary);
  std::ofstream meta(base + "/draft.txt");
  if (!features_out || !logits_out || !meta)
    return 1;
  features_out.write((const char*)host.data(),
                     (std::size_t)rows * DS41_DSPARK_TARGET_LAYERS * 5120 * 4);
  logits_out.write((const char*)draft_logits.data(), draft_logits.size() * 4);
  meta << pos0 << " " << rows << " " << anchor;
  for (int32_t t : drafts)
    meta << " " << t;
  meta << "\n";
  features_out.close();
  logits_out.close();
  meta.close();
  if (!features_out || !logits_out || !meta)
    return 1;
  std::printf("dumped %u feature rows from %u, anchor %d, drafts", rows, pos0,
              anchor);
  for (int32_t t : drafts)
    std::printf(" %d", t);
  std::printf("\n");
  // The shortened Markov chain must produce the original prefix and must
  // not overwrite the caller's unused output slots.
  for (uint32_t width = 1; width < DS41_DSPARK_BLOCK; width++) {
    int32_t prefix[DS41_DSPARK_BLOCK];
    std::fill_n(prefix, DS41_DSPARK_BLOCK, -1);
    if (!ds41_dspark_draft(s.get(), anchor, prefix, width))
      return 1;
    if (!ds41_dspark_copy_logits(engine, comparison_logits.data()) ||
        std::memcmp(comparison_logits.data(), draft_logits.data(),
                    (std::size_t)width * logits.size() * sizeof(float)) != 0) {
      std::fprintf(stderr, "draft head logits differ at width %u\n", width);
      return 1;
    }
    for (uint32_t i = 0; i < DS41_DSPARK_BLOCK; i++) {
      if (prefix[i] != (i < width ? drafts[i] : -1)) {
        std::fprintf(stderr, "draft prefix mismatch: width %u slot %u\n", width,
                     i);
        return 1;
      }
    }
  }
  if (ds41_dspark_draft(s.get(), anchor, drafts, 0) ||
      ds41_dspark_draft(s.get(), anchor, drafts, DS41_DSPARK_BLOCK + 1))
    return 1;
  std::printf(
      "draft prefixes/logits 1..5 identical; output tails untouched; invalid "
      "widths "
      "rejected\n");
  /* The same draft in batches: sessions over shorter prefixes of the prompt
   * draft with it, before and after it, and its drafts and head logits must
   * match its own draft's. */
  std::fprintf(stderr, "batched draft check\n");
  std::vector<int> anchors;
  anchors.reserve(kOthers);
  for (uint32_t o = 0; o < kOthers; ++o) {
    const uint32_t len = (uint32_t)prompt.size() - 3 * (o + 1);
    for (uint32_t at = 0; at < len;) {
      const uint32_t n = std::min<uint32_t>(cap, len - at);
      uint32_t fp = 0, fr = 0;
      const int rc = ds41_decoder_eval(od[o].get(), prompt.data() + at, n,
                                       logits.data(), 1, nullptr, nullptr);
      if (rc != 0 || !ds41_decoder_features(od[o].get(), &fp, &fr)) {
        std::fprintf(stderr,
                     "batched draft: session %u prefill failed (rc %d, at %u "
                     "of %u)\n",
                     o, rc, at, len);
        return 1;
      }
      ds41_dspark_state_reset(os[o].get(), fp);
      if (!ds41_dspark_inject(os[o].get(), fp, fr)) {
        std::fprintf(stderr, "batched draft: session %u inject failed\n", o);
        return 1;
      }
      at += n;
    }
    anchors.push_back(Argmax(logits.data(), logits.size()));
  }
  for (const bool last : {false, true}) {
    ds41_dspark_draft_request req[kOthers + 1];
    int32_t out[kOthers + 1][DS41_DSPARK_BLOCK];
    const uint32_t mine = last ? kOthers : 0;
    for (uint32_t i = 0, o = 0; i <= kOthers; i++) {
      req[i].drafts = out[i];
      req[i].count = DS41_DSPARK_BLOCK;
      if (i == mine) {
        req[i].state = s.get();
        req[i].anchor = anchor;
      } else {
        req[i].state = os[o].get();
        req[i].anchor = anchors[o++];
      }
    }
    if (!ds41_dspark_draft_batch(req, kOthers + 1) || !req[mine].drafted) {
      std::fprintf(stderr, "batched draft failed\n");
      return 1;
    }
    if (!ds41_dspark_copy_logits(engine, comparison_logits.data(),
                                 mine * DS41_DSPARK_BLOCK, DS41_DSPARK_BLOCK)) {
      std::fprintf(stderr, "batched draft logits not copied\n");
      return 1;
    }
    for (uint32_t r = 0; r < DS41_DSPARK_BLOCK; r++) {
      const std::size_t v = logits.size();
      if (std::memcmp(comparison_logits.data() + r * v,
                      draft_logits.data() + r * v, v * sizeof(float)) != 0) {
        float diff = 0.0f;
        for (std::size_t i = 0; i < v; i++)
          diff = std::max(diff, std::fabs(comparison_logits[r * v + i] -
                                          draft_logits[r * v + i]));
        std::fprintf(stderr,
                     "batched draft (%s) head logits differ at row %u: max "
                     "%g\n",
                     last ? "last" : "first", r, diff);
        return 1;
      }
      if (out[mine][r] != drafts[r]) {
        std::fprintf(stderr, "batched draft (%s) differs at slot %u\n",
                     last ? "last" : "first", r);
        return 1;
      }
    }
  }
  std::printf("batched drafts (first and last of %u) identical\n", kOthers + 1);
  return 0;
}

/* Decoder SWA bounded replay against the exact prefill: the ids but the
 * last `rows` are the prompt, prefilled once in full and once with
 * ds41_decoder_eval_prompt; the last `rows` are then evaluated as one
 * batch (teacher forcing) after each. Reports the prefill times, the
 * reference tokens' NLL under both, top-1 agreement and KL(full || replay)
 * per continuation position. */
int ReplayProbe(ds41_engine* engine, const std::vector<int>& ids,
                uint32_t context, uint32_t rows, ProbeStatus* status) {
  if (rows < 2 || std::uint64_t{rows} + 2 > ids.size())
    return 2;
  const uint32_t prompt = (uint32_t)(ids.size() - rows);
  DecoderPtr d(
      ds41_decoder_create(
          engine, std::max<uint32_t>(context, (uint32_t)ids.size() + 1)),
      &ds41_decoder_destroy);
  if (!d)
    return 1;
  const std::size_t vocab = (std::size_t)ds41_engine_vocab_size(engine);
  if (vocab == 0 || rows > std::numeric_limits<std::size_t>::max() / vocab)
    return 1;
  const uint32_t cap = ds41_decoder_batch_capacity(d.get());
  if (rows > cap)
    return 2;
  std::vector<float> logits[2];
  logits[0].resize((std::size_t)rows * vocab);
  logits[1].resize((std::size_t)rows * vocab);
  std::vector<float> last(vocab);
  double prefill_ms[2] = {};
  const int fit = FitStreamedExperts(engine, status, true);
  if (fit != 0)
    return fit;
  for (int replay = 0; replay < 2; ++replay) {
    if (ds41_decoder_reset(d.get()) != 0)
      return 1;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < prompt;) {
      const uint32_t n = std::min(cap, prompt - i);
      const int rc = replay
                         ? ds41_decoder_eval_prompt(d.get(), ids.data() + i, n,
                                                    prompt, last.data(), 0)
                         : ds41_decoder_eval(d.get(), ids.data() + i, n,
                                             last.data(), 0, nullptr, nullptr);
      if (rc != 0) {
        std::fprintf(stderr, "prefill failed at %u\n", i);
        return 1;
      }
      i += n;
    }
    prefill_ms[replay] = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    /* Row r predicts ids[prompt + r]: the batch is the prompt's last token
     * and the continuation but its last token. */
    if (ds41_decoder_reset(d.get()) != 0)
      return 1;
    /* Re-run the prompt without its last token, then the scored batch. */
    for (uint32_t i = 0; i + 1 < prompt;) {
      const uint32_t n = std::min(cap, prompt - 1 - i);
      const int rc = replay
                         ? ds41_decoder_eval_prompt(d.get(), ids.data() + i, n,
                                                    prompt - 1, last.data(), 0)
                         : ds41_decoder_eval(d.get(), ids.data() + i, n,
                                             last.data(), 0, nullptr, nullptr);
      if (rc != 0)
        return 1;
      i += n;
    }
    if (ds41_decoder_eval(d.get(), ids.data() + prompt - 1, rows,
                          logits[replay].data(), rows, nullptr, nullptr) != 0)
      return 1;
  }
  double nll[2] = {}, kl_sum = 0.0, kl_max = 0.0;
  uint32_t agree = 0, ref_hits[2] = {};
  for (uint32_t r = 0; r < rows; ++r) {
    const int target = ids[prompt + r];
    if (target < 0 || (std::size_t)target >= vocab)
      return 2;
    double lse[2];
    int best[2];
    for (int k = 0; k < 2; ++k) {
      const float* row = logits[k].data() + (std::size_t)r * vocab;
      best[k] = Argmax(row, vocab);
      double sum = 0.0;
      for (std::size_t i = 0; i < vocab; ++i)
        sum += std::exp((double)row[i] - row[best[k]]);
      lse[k] = row[best[k]] + std::log(sum);
      nll[k] += lse[k] - row[target];
      ref_hits[k] += best[k] == target;
    }
    agree += best[0] == best[1];
    const float* p = logits[0].data() + (std::size_t)r * vocab;
    const float* q = logits[1].data() + (std::size_t)r * vocab;
    double kl = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
      const double lp = p[i] - lse[0], lq = q[i] - lse[1];
      kl += std::exp(lp) * (lp - lq);
    }
    kl_sum += kl;
    kl_max = std::max(kl_max, kl);
  }
  std::printf(
      "replay prompt %u tokens: prefill %.1f ms full, %.1f ms replay "
      "(%.2fx)\n",
      prompt, prefill_ms[0], prefill_ms[1], prefill_ms[0] / prefill_ms[1]);
  std::printf(
      "replay continuation %u tokens: nll %.4f full, %.4f replay; "
      "top-1 reference %u / %u; agreement %u; kl mean %.5f max %.4f\n",
      rows, nll[0] / rows, nll[1] / rows, ref_hits[0], ref_hits[1], agree,
      kl_sum / rows, kl_max);
  return 0;
}

int RunProbe(int argc, char** argv, ProbeStatus& status) {
  const char* model = nullptr;
  const char* tokenize = nullptr;
  const char* decode = nullptr;
  const char* out_dir = nullptr;
  int generate = 0;
  uint32_t context = 0;
  uint32_t prefill_batch = 0;
  bool score = false;
  bool batch = false;
  bool upload = false;
  bool rendered = false;
  int spec_check = 0;
  int spec_width = 4;
  int spec_bench = -1;
  int joint_check = 0;
  int joint_cycles = 24;
  uint32_t joint_rows = DS41_DECODER_MAX_JOINT;
  const char* dspark = nullptr;
  int dspark_generate = 0;
  const char* dspark_dump = nullptr;
  bool dspark_sweep = false;
  uint32_t replay_check = 0;
  bool exact_prefill = false;
  bool stream_experts = false;
  uint64_t expert_cache_bytes = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--status-json") == 0) {
      if (i + 1 == argc)
        return 2;
      status.path = argv[++i];
    } else if (std::strcmp(argv[i], "--tokenize") == 0 && i + 1 < argc) {
      tokenize = argv[++i];
    } else if (std::strcmp(argv[i], "--decode") == 0 && i + 1 < argc) {
      decode = argv[++i];
      upload = true;
    } else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
      generate = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
      context = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
    } else if (std::strcmp(argv[i], "--prefill-batch") == 0 && i + 1 < argc) {
      prefill_batch = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
    } else if (std::strcmp(argv[i], "--score") == 0) {
      score = true;
    } else if (std::strcmp(argv[i], "--batch") == 0) {
      batch = true;
    } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (std::strcmp(argv[i], "--spec-check") == 0 && i + 1 < argc) {
      spec_check = std::atoi(argv[++i]);
      upload = true;
    } else if (std::strcmp(argv[i], "--joint-check") == 0 && i + 1 < argc) {
      joint_check = std::atoi(argv[++i]);
      upload = true;
    } else if (std::strcmp(argv[i], "--joint-cycles") == 0 && i + 1 < argc) {
      joint_cycles = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--joint-rows") == 0 && i + 1 < argc) {
      joint_rows =
          std::clamp<uint32_t>((uint32_t)std::strtoul(argv[++i], nullptr, 10),
                               1, DS41_DECODER_MAX_JOINT);
    } else if (std::strcmp(argv[i], "--spec-width") == 0 && i + 1 < argc) {
      spec_width =
          std::clamp(std::atoi(argv[++i]), 1, DS41_DECODER_MAX_SPECULATIVE);
    } else if (std::strcmp(argv[i], "--spec-bench") == 0 && i + 1 < argc) {
      spec_bench = std::atoi(argv[++i]);
      upload = true;
    } else if (std::strcmp(argv[i], "--dspark") == 0 && i + 1 < argc) {
      dspark = argv[++i];
    } else if (std::strcmp(argv[i], "--dspark-generate") == 0 && i + 1 < argc) {
      dspark_generate = std::atoi(argv[++i]);
      upload = true;
    } else if (std::strcmp(argv[i], "--dspark-sweep") == 0) {
      dspark_sweep = true;
    } else if (std::strcmp(argv[i], "--dspark-dump") == 0 && i + 1 < argc) {
      dspark_dump = argv[++i];
      upload = true;
    } else if (std::strcmp(argv[i], "--replay-check") == 0 && i + 1 < argc) {
      replay_check = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
      upload = true;
    } else if (std::strcmp(argv[i], "--exact-prefill") == 0) {
      exact_prefill = true;
    } else if (std::strcmp(argv[i], "--stream-experts") == 0) {
      stream_experts = true;
    } else if (std::strcmp(argv[i], "--expert-cache-bytes") == 0 &&
               i + 1 < argc) {
      expert_cache_bytes = std::strtoull(argv[++i], nullptr, 10);
    } else if (std::strcmp(argv[i], "--rendered") == 0) {
      rendered = true;
    } else if (std::strcmp(argv[i], "--upload") == 0) {
      upload = true;
    } else if (!model) {
      model = argv[i];
    } else {
      model = nullptr;
      break;
    }
  }
  if (!model || generate < 0) {
    std::fprintf(
        stderr,
        "usage: %s MODEL-00001-of-N.gguf [--upload] [--status-json FILE] "
        "[--tokenize TEXTS [--rendered]] [--decode IDS [--generate N] "
        "[--score] [--batch] [--context N] [--prefill-batch N] "
        "[--out DIR]] [--replay-check ROWS] [--exact-prefill] "
        "[--stream-experts [--expert-cache-bytes N]] "
        "[--spec-check CYCLES [--spec-width W] | --spec-bench N | "
        "--joint-check SESSIONS [--joint-cycles N] [--joint-rows N] | "
        "--dspark FILE "
        "--dspark-generate N [--dspark-sweep] | --dspark FILE --dspark-dump "
        "DIR]\n",
        argv[0]);
    return 2;
  }
  Report("start");
  const auto begin = std::chrono::steady_clock::now();
  ds41_engine* engine = nullptr;
  std::unique_ptr<ds41_engine, EngineCloser> engine_owner(
      nullptr, EngineCloser{&status});
  DecoderPtr decoder(nullptr, &ds41_decoder_destroy);
  auto close = [&](int result) {
    decoder.reset();
    engine_owner.reset();
    engine = nullptr;
    Report("closed");
    return result;
  };
  auto transition = [&](Stage stage, std::uint64_t completed,
                        std::uint64_t total, const char* failure = nullptr) {
    const int result =
        status.stages.Transition(stage, completed, total, failure);
    return result != 0 ? close(result) : result;
  };
  ds41_engine_options options{};
  options.model_path = model;
  options.validate_only = !upload;
  options.prefill_batch = prefill_batch;
  options.exact_prefill = exact_prefill;
  options.stream_experts = stream_experts;
  options.expert_cache_bytes = expert_cache_bytes;
  options.dspark_model_path = dspark;
  status.upload = upload;
  status.requested_streaming = stream_experts;
  status.requested_cache_bytes = expert_cache_bytes;
  status.ordinary = decode && !replay_check && !spec_check && spec_bench < 0 &&
                    joint_check <= 0 && dspark_generate <= 0 && !dspark_dump;
  status.stages = ProbeStages(tokenize != nullptr);
  if (decode && !*decode)
    return transition(Stage::kPrefill, 0, 0, "empty_decode_prompt");
  if (const int result = transition(Stage::kLoading, 0, 1); result != 0)
    return result;
  if (ds41_engine_open(&engine, &options) != 0)
    return transition(Stage::kLoading, 0, 1, "engine_open_failed");
  engine_owner.reset(engine);
  if (const int result = transition(Stage::kLoading, 1, 1); result != 0)
    return result;
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
  std::printf("opened %s in %.1f s (vocab %d)\n",
              ds41_engine_model_name(engine), seconds,
              ds41_engine_vocab_size(engine));
  Report("loaded");
  if (tokenize) {
    /* One text per line; "\n" in a line stands for a newline. Prints the ids.
     */
    std::ifstream in(tokenize);
    if (const int result = transition(Stage::kTokenization, 0, 1); result != 0)
      return result;
    std::string line;
    while (std::getline(in, line)) {
      std::string text;
      for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == 'n') {
          text += '\n';
          ++i;
        } else {
          text += line[i];
        }
      }
      ds41_tokens tokens{};
      if (rendered)
        ds41_tokenize_rendered_chat(engine, text.c_str(), &tokens);
      else
        ds41_tokenize_text(engine, text.c_str(), &tokens);
      ++status.tokenized_lines;
      status.tokenized_tokens += tokens.len;
      std::printf("ids");
      for (int i = 0; i < tokens.len; ++i)
        std::printf(" %d", tokens.v[i]);
      std::printf("\n");
      ds41_tokens_free(&tokens);
    }
    const bool tokenized =
        in.eof() && status.tokenized_lines > 0 && status.tokenized_tokens > 0;
    if (const int result =
            transition(Stage::kTokenization, tokenized ? 1 : 0, 1,
                       tokenized ? nullptr : "tokenization_failed");
        result != 0)
      return result;
  }
  if (decode && !status.ordinary) {
    if (const int result = transition(Stage::kPrefill, 0, 1); result != 0)
      return result;
  }
  if (decode && replay_check) {
    std::vector<int> ids;
    for (const char* p = decode; *p;) {
      ids.push_back(std::atoi(p));
      while (*p && *p != ',')
        ++p;
      if (*p == ',')
        ++p;
    }
    return close(ReplayProbe(engine, ids, context, replay_check, &status));
  }
  if (decode && (spec_check || spec_bench >= 0 || joint_check > 0 ||
                 dspark_generate > 0 || dspark_dump)) {
    std::vector<int> prompt;
    for (const char* p = decode; *p;) {
      prompt.push_back(std::atoi(p));
      while (*p && *p != ',')
        ++p;
      if (*p == ',')
        ++p;
    }
    if (dspark_dump)
      return close(DsparkDump(engine, prompt, dspark_dump, &status));
    if (joint_check > 0)
      return close(
          JointProbe(engine, prompt, context, joint_check, joint_cycles,
                     std::max<uint32_t>(joint_rows, joint_check), &status));
    if (dspark_generate > 0)
      return close(DsparkProbe(engine, prompt, context, dspark_generate,
                               dspark_sweep, &status));
    return close(SpeculativeProbe(engine, prompt, context, spec_check,
                                  spec_bench, spec_width, &status));
  }
  if (decode) {
    /* Feed comma-separated ids one at a time (--batch: in decoder batches),
     * then decode greedily. */
    std::vector<int> tokens;
    for (const char* p = decode; *p;) {
      tokens.push_back(std::atoi(p));
      while (*p && *p != ',')
        ++p;
      if (*p == ',')
        ++p;
    }
    const uint32_t total = (uint32_t)tokens.size() + (uint32_t)generate;
    if (const int result = transition(Stage::kPrefill, 0, tokens.size());
        result != 0)
      return result;
    decoder =
        DecoderPtr(ds41_decoder_create(engine, std::max(context, total + 1)),
                   &ds41_decoder_destroy);
    if (!decoder)
      return transition(Stage::kPrefill, 0, tokens.size(),
                        "decoder_create_failed");
    Report("decoder");
    const size_t vocab = (size_t)ds41_engine_vocab_size(engine);
    for (int token : tokens) {
      if (token < 0 || (size_t)token >= vocab)
        return transition(Stage::kPrefill, 0, tokens.size(),
                          "token_out_of_range");
    }
    Dump dump;
    dump.dir = out_dir ? out_dir : "";
    const bool want_dump = out_dir && !score;
    const bool want_score = out_dir && score;
    FilePtr routes_file(
        want_dump
            ? std::fopen((std::string(out_dir) + "/routes.txt").c_str(), "w")
            : nullptr);
    /* --score writes the argmax and next-token NLL per prompt position instead
     * of logits. */
    FilePtr logits_file(
        want_dump ? std::fopen((dump.dir + "/logits.f32").c_str(), "wb")
                  : nullptr);
    FilePtr score_file(want_score
                           ? std::fopen((dump.dir + "/score.txt").c_str(), "w")
                           : nullptr);
    if ((want_dump && (!routes_file || !logits_file)) ||
        (want_score && !score_file))
      return transition(Stage::kPrefill, 0, tokens.size(),
                        "output_open_failed");
    dump.routes = routes_file.get();
    std::vector<float> logits(vocab);
    std::vector<int> generated;
    generated.reserve(static_cast<std::size_t>(generate) + 1);
    const uint32_t cap = ds41_decoder_batch_capacity(decoder.get());
    std::vector<float> rows(batch ? static_cast<size_t>(cap) * vocab : 0);
    // The decoder and probe buffers exist before the shared cache-fit step.
    const int fit = FitStreamedExperts(engine, &status, false);
    if (fit != 0)
      return close(fit);
    /* Records the logits of position `step` (input `token`). */
    auto consume = [&](uint32_t step, int token, const float* row, double ms) {
      if (logits_file)
        std::fwrite(row, sizeof(float), vocab, logits_file.get());
      int best = 0;
      for (size_t i = 1; i < vocab; ++i)
        if (row[i] > row[(size_t)best])
          best = (int)i;
      if (step + 1 >= tokens.size())
        generated.push_back(best);
      if (score_file && step + 1 < tokens.size()) {
        double sum = 0.0;
        for (size_t i = 0; i < vocab; ++i)
          sum += std::exp((double)row[i] - (double)row[(size_t)best]);
        const double nll =
            std::log(sum) -
            ((double)row[(size_t)tokens[step + 1]] - (double)row[(size_t)best]);
        std::fprintf(score_file.get(), "%u %d %d %.6f %.2f\n", step, best,
                     tokens[step + 1], nll, ms);
      } else if (!score_file) {
        std::printf("step %u token %d -> %d (%.1f ms)\n", step, token, best,
                    ms);
      }
    };
    uint32_t step = 0;
    if (batch) {
      /* The prompt in decoder-sized batches with logits for every position. */
      while (step < tokens.size()) {
        const uint32_t n =
            std::min<uint32_t>(cap, (uint32_t)tokens.size() - step);
        const auto t0 = std::chrono::steady_clock::now();
        /* Every row's logits only when they are recorded. */
        const uint32_t logit_rows = logits_file || score_file ? n : 1;
        if (ds41_decoder_eval(decoder.get(), tokens.data() + step, n,
                              rows.data(), logit_rows, nullptr, nullptr) != 0) {
          std::fprintf(stderr, "batch failed at step %u\n", step);
          return transition(Stage::kPrefill, step, tokens.size(),
                            "prefill_failed");
        }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        std::fprintf(stderr, "batch %u tokens at %u: %.1f ms (%.1f tok/s)\n", n,
                     step, ms, n * 1000.0 / ms);
        for (uint32_t i = n - logit_rows; i < n; ++i)
          consume(step + i, tokens[step + i],
                  rows.data() + (size_t)(i - (n - logit_rows)) * vocab, ms / n);
        step += n;
        if (const int result = transition(Stage::kPrefill, step, tokens.size());
            result != 0)
          return result;
      }
    }
    for (; step < total; ++step) {
      const bool is_prompt = step < tokens.size();
      if (!is_prompt) {
        if (const int result =
                transition(Stage::kDecode, step - tokens.size(), generate);
            result != 0)
          return result;
      }
      const int token = step < tokens.size() ? tokens[step] : generated.back();
      dump.pos = step;
      const auto t0 = std::chrono::steady_clock::now();
      if (ds41_decoder_step(decoder.get(), token, logits.data(),
                            out_dir && !score ? DumpLayer : nullptr,
                            &dump) != 0) {
        std::fprintf(stderr, "decode failed at step %u\n", step);
        return transition(is_prompt ? Stage::kPrefill : Stage::kDecode, 0, 1,
                          is_prompt ? "prefill_failed" : "decode_failed");
      }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      consume(step, token, logits.data(), ms);
      const int result =
          is_prompt
              ? transition(Stage::kPrefill, step + 1, tokens.size())
              : transition(Stage::kDecode, step + 1 - tokens.size(), generate);
      if (result != 0)
        return result;
    }
    std::printf("generated");
    for (size_t i = 0;
         i + 1 < generated.size() || (i < generated.size() && generate > 0);
         ++i)
      std::printf(" %d", generated[i]);
    std::printf("\n");
    Report("evaluated");
  }
  status.expert_cache = ds41_engine_expert_cache_status(engine);
  status.expert_cache_read = true;
  return close(0);
}

int main(int argc, char** argv) {
  ProbeStatus status;
  int result = 1;
  try {
    result = RunProbe(argc, argv, status);
  } catch (const std::exception& error) {
    result = status.stages.Transition(Stage::kActive, 0, 1, "probe_exception");
    std::fprintf(stderr, "Probe failed: %s\n", error.what());
  }
  return status.Write(result) ? result : 1;
}
