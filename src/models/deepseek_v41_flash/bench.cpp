// Single-session, depth-zero autoregressive benchmark from franzmoca/gufo
// 01fb6f4e39464003efffd97d79607005146280f0, src/cli/bench/bench.cpp.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "gufo/ext/crypto.hpp"
#include "src/models/deepseek_v41_flash/engine.hpp"
#include "src/models/deepseek_v41_flash/runner.hpp"

namespace gufo::models::deepseek_v41_flash {
namespace {

struct BenchStats {
  double mean{0.0};
  double stddev{0.0};
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty())
    return {};
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  if (values.size() <= 1)
    return {.mean = mean, .stddev = 0.0};
  double sq_sum = 0.0;
  for (const double value : values)
    sq_sum += (value - mean) * (value - mean);
  return {.mean = mean,
          .stddev = std::sqrt(sq_sum / static_cast<double>(values.size() - 1))};
}

std::vector<int> MakeDeepSeekBenchmarkTokens(const Model& model,
                                             std::size_t count) {
  const auto pattern = model.Tokenize(
      "The quick brown fox jumps over the lazy dog. "
      "Strix Halo executes this deterministic benchmark sequence. ");
  if (pattern.empty())
    throw std::runtime_error("DeepSeek benchmark token pattern is empty");
  std::vector<int> tokens(count);
  for (std::size_t index = 0; index < count; ++index)
    tokens[index] = pattern[index % pattern.size()];
  return tokens;
}

void PrintExpertCache(const Model& model) {
  const auto cache = model.ExpertCacheStatus();
  std::cerr << "[Expert Cache]: requested_bytes=" << cache.requested_bytes
            << " allocated_bytes=" << cache.allocated_bytes
            << " limit_bytes=" << cache.limit_bytes << '\n';
}

}  // namespace

int RunDeepSeekBenchmark(const ext::TextBenchOptions& options,
                         const std::shared_ptr<const core::GgufReader>& reader,
                         std::chrono::steady_clock::time_point load_start) {
  if (options.concurrency != std::vector<std::size_t>{1} ||
      options.n_depths != std::vector<std::size_t>{0}) {
    std::cerr << "Error: DeepSeek V4.1 bench requires --concurrency 1 "
                 "and --n-depth 0\n";
    return 1;
  }
  if (!options.speculative_backend.empty() ||
      !options.dspark_model_path.empty() || !options.mtp_model_path.empty() ||
      !options.dflash_model_path.empty() || !options.draft_policy.empty() ||
      options.draft_tokens != 7 || options.min_draft_tokens != 1) {
    std::cerr << "Error: DeepSeek V4.1 bench does not support speculative "
                 "decoding\n";
    return 1;
  }
  if (!options.sampling.can_use_unmodified_argmax()) {
    std::cerr << "Error: DeepSeek V4.1 bench requires greedy sampling\n";
    return 1;
  }
  if (options.enable_yarn || options.validate_prefill_tokens != 0) {
    std::cerr << "Error: DeepSeek V4.1 bench does not support --enable-yarn "
                 "or --validate-prefill\n";
    return 1;
  }
  if (options.expert_cache_bytes != 0 && !options.stream_experts) {
    std::cerr << "Error: --expert-cache-bytes requires --stream-experts\n";
    return 1;
  }
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const auto max_prompt = max_or_zero(options.n_prompts);
  const auto max_generation = max_or_zero(options.n_gens);
  constexpr std::size_t context_limit = std::numeric_limits<int>::max() - 1;
  if (max_prompt > context_limit || max_generation > context_limit - 16) {
    std::cerr << "Error: requested DeepSeek benchmark context is too large\n";
    return 1;
  }
  const auto required_context = static_cast<std::uint32_t>(
      std::max({std::size_t{4096}, max_prompt + 1, 16 + max_generation + 1}));
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for DeepSeek V4.1 Flash\n";
    return 1;
  }

  std::string error;
  auto model =
      Model::Load(options.model_path,
                  ModelOptions{.max_context = required_context,
                               .stream_experts = options.stream_experts,
                               .expert_cache_bytes = options.expert_cache_bytes,
                               .dspark_model_path = ""},
                  &error);
  if (!model) {
    std::cerr << "Error creating DeepSeek V4.1 Flash model: " << error << '\n';
    return 1;
  }
  std::cout << "[Model Load]: "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             load_start)
                   .count()
            << " s\n";
  PrintExpertCache(*model);
  const auto tokens = MakeDeepSeekBenchmarkTokens(*model, required_context);
  const double size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  const auto print_result = [&](std::string_view test,
                                const BenchStats& stats) {
    std::cout << "| " << model->ModelName() << " | " << std::fixed
              << std::setprecision(2) << size_gib << " GiB | ROCm (HIP) | "
              << test << " | " << stats.mean << " ± " << stats.stddev << " |\n"
              << std::flush;
  };
  std::cout << "| model | size | backend | test | t/s |\n"
               "| --- | --- | --- | --- | --- |\n";

  // Preserve the source order: all prefill repetitions, then all decode
  // repetitions. The model and expert store remain alive across sessions.
  for (const auto prompt_length : options.n_prompts) {
    std::vector<double> runs;
    runs.reserve(options.repetitions);
    for (std::size_t repetition = 0; repetition < options.repetitions;
         ++repetition) {
      auto session = model->CreateSession(core::SessionMode::kAutoregressive,
                                          required_context, &error);
      if (!session) {
        std::cerr << "Error creating DeepSeek session: " << error << '\n';
        return 1;
      }
      const auto start = std::chrono::steady_clock::now();
      if (!session->Sync(std::span(tokens).first(prompt_length), &error)) {
        std::cerr << "Error running DeepSeek prefill: " << error << '\n';
        return 1;
      }
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      runs.push_back(static_cast<double>(prompt_length) / seconds);
      if (options.verbose)
        std::cerr << "DeepSeek pp depth=0 tok/s=" << runs.back()
                  << " payload_bytes=" << session->PayloadBytes() << '\n';
    }
    print_result("pp" + std::to_string(prompt_length), ComputeStats(runs));
  }

  for (const auto generation_length : options.n_gens) {
    std::vector<double> runs;
    runs.reserve(options.repetitions);
    for (std::size_t repetition = 0; repetition < options.repetitions;
         ++repetition) {
      auto session = model->CreateSession(core::SessionMode::kAutoregressive,
                                          required_context, &error);
      if (!session || !session->Sync(std::span(tokens).first(16), &error)) {
        std::cerr << "Error preparing DeepSeek generation: " << error << '\n';
        return 1;
      }
      std::vector<int> generated;
      generated.reserve(generation_length);
      const auto start = std::chrono::steady_clock::now();
      for (std::size_t step = 0; step < generation_length; ++step) {
        const int token = session->SelectNext(0.0F, nullptr);
        if (token < 0 || !session->Evaluate(token, &error)) {
          std::cerr << "Error running DeepSeek decode: " << error << '\n';
          return 1;
        }
        generated.push_back(token);
      }
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      runs.push_back(static_cast<double>(generation_length) / seconds);
      if (options.verbose) {
        const auto* bytes =
            reinterpret_cast<const std::uint8_t*>(generated.data());
        std::cerr << "DeepSeek tg C=1 depth=0 request=0 output_sha256="
                  << crypto::Sha256Hex(
                         std::span(bytes, generated.size() * sizeof(int)))
                  << '\n';
      }
    }
    print_result("tg" + std::to_string(generation_length), ComputeStats(runs));
  }
  PrintExpertCache(*model);
  return 0;
}

}  // namespace gufo::models::deepseek_v41_flash
