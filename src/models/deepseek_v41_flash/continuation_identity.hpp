#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_CONTINUATION_IDENTITY_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_CONTINUATION_IDENTITY_HPP_

#include <cstdint>
#include <string_view>
#include <vector>

#include "src/models/deepseek_v41_flash/runtime/variant.h"

namespace DS41_VARIANT_NAMESPACE {

std::vector<std::uint8_t> DeepSeek41CompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view support_fingerprint,
    std::uint32_t max_context, bool use_dspark, std::uint32_t max_draft_tokens);

// Identity of a loaded model that was given no file fingerprints. It is valid
// only inside this process, so it can never match a durable cache.
std::vector<std::uint8_t> DeepSeek41ProcessIdentity(
    const void* model, std::uint32_t max_context, bool use_dspark,
    std::uint32_t max_draft_tokens);

}  // namespace DS41_VARIANT_NAMESPACE

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_CONTINUATION_IDENTITY_HPP_
