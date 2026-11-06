#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_DSPARK_SAMPLER_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_DSPARK_SAMPLER_HPP_

#include <cstdint>
#include <gufo/ext/sampling.hpp>
#include <span>
#include <utility>
#include <vector>

#include "src/models/deepseek_v41_flash/runtime/model.h"

DS41_NAMESPACE_BEGIN

/* Lends a request sampler to V4.1 DSpark cycles. The engine's greedy path
 * stays available; a sampled request draws through this bridge and keeps the
 * request's filters, grammar, penalties, RNG and logprobs. */
class DsparkSamplerBridge {
public:
  explicit DsparkSamplerBridge(const sampling::SamplerState& sampler)
      : working_(sampler),
        hook_{
            .ctx = this,
            .sample = &Sample,
            .accept = &Accept,
            .try_accept_argmax = !sampler.config().logprobs &&
                                         sampler.config().temperature == 0.F &&
                                         !sampler.config().penalties_enabled()
                                     ? &TryAcceptArgmax
                                     : nullptr,
            .unmodified_argmax = sampler.config().can_use_unmodified_argmax()} {
  }

  DsparkSamplerBridge(const DsparkSamplerBridge&) = delete;
  DsparkSamplerBridge& operator=(const DsparkSamplerBridge&) = delete;

  [[nodiscard]] const ds41_dspark_sampler* hook() const { return &hook_; }
  [[nodiscard]] std::uint64_t rng_state() const { return working_.rng_state(); }

  [[nodiscard]] const std::vector<sampling::TokenLogprobs>& logprobs() const {
    return logprobs_;
  }

private:
  static void Observe(DsparkSamplerBridge* self, const float* logits,
                      std::uint32_t size, int token) {
    if (auto value = sampling::CollectLogprobs(
            {logits, size}, token, self->working_.config().logprobs))
      self->logprobs_.push_back(std::move(*value));
  }

  static int Sample(void* ctx, const float* logits,
                    std::uint32_t vocabulary_size) {
    auto& self = *static_cast<DsparkSamplerBridge*>(ctx);
    const auto token =
        self.working_.Sample(std::span<const float>(logits, vocabulary_size));
    Observe(&self, logits, vocabulary_size, static_cast<int>(token));
    self.working_.Accept(token);
    return static_cast<int>(token);
  }

  static void Accept(void* ctx, int token) {
    static_cast<DsparkSamplerBridge*>(ctx)->working_.Accept(
        static_cast<sampling::TokenId>(token));
  }

  static bool TryAcceptArgmax(void* ctx, int token) {
    auto& working = static_cast<DsparkSamplerBridge*>(ctx)->working_;
    if (token < 0 ||
        !working.CanSelectArgmax(static_cast<sampling::TokenId>(token)))
      return false;
    working.Accept(static_cast<sampling::TokenId>(token));
    return true;
  }

  sampling::SamplerState working_;
  std::vector<sampling::TokenLogprobs> logprobs_;
  ds41_dspark_sampler hook_;
};

DS41_NAMESPACE_END

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_DSPARK_SAMPLER_HPP_