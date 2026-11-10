#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_OPTIONS_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_OPTIONS_HPP_

#include <cstdint>
#include <gufo/ext/options.hpp>
#include <gufo/ext/text.hpp>
#include <stdexcept>

namespace gufo::models::deepseek_v41_flash {

struct ExpertOptions {
  bool stream_experts{false};
  std::uint64_t expert_cache_bytes{0};
};

inline ExpertOptions ResolveExpertOptions(const ext::ParsedOptions& options) {
  ExpertOptions result;
  if (const auto* value = ext::FindOption<bool>(options, "--stream-experts"))
    result.stream_experts = *value;
  if (const auto* value =
          ext::FindOption<std::int64_t>(options, "--expert-cache-bytes")) {
    if (*value < 0)
      throw std::invalid_argument("--expert-cache-bytes must not be negative");
    result.expert_cache_bytes = static_cast<std::uint64_t>(*value);
  }
  if (result.expert_cache_bytes != 0 && !result.stream_experts)
    throw std::invalid_argument(
        "--expert-cache-bytes requires --stream-experts");
  return result;
}

inline sampling::TextSamplingDefaults DeepSeekSamplingDefaults() {
  sampling::TextSamplingValues values;
  values.temperature = 1.0F;
  values.top_p = 1.0F;
  return {values, values, values};
}

}  // namespace gufo::models::deepseek_v41_flash

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_OPTIONS_HPP_
