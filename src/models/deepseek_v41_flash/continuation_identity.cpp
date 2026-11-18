#include "src/models/deepseek_v41_flash/continuation_identity.hpp"

#include <algorithm>
#include <gufo/ext/continuation.hpp>
#include <sstream>
#include <stdexcept>

#include "src/models/deepseek_v41_flash/chat_template.hpp"
#include "src/models/deepseek_v41_flash/continuation_layout.hpp"
#include "src/models/deepseek_v41_flash/runtime/variant.h"

namespace DS41_VARIANT_NAMESPACE {

static bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::vector<std::uint8_t> DeepSeek41CompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view support_fingerprint,
    std::uint32_t max_context, bool use_dspark,
    std::uint32_t max_draft_tokens) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "DeepSeek V4.1 disk cache requires an artifact fingerprint");
  }
  if (use_dspark && !IsSha256Hex(support_fingerprint)) {
    throw std::invalid_argument(
        "DeepSeek V4.1 DSpark disk cache requires a drafter fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=gufo-deepseek41-continuation-v1\n"
           << "model_kind=deepseek41\n"
           << "variant=" << DS41_MODEL_SHAPE_NAME << '\n'
           << "expert_count=" << DS41_VARIANT_EXPERT_COUNT << '\n'
           << "artifact_id=" << ext::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=artifact-embedded-v1\n"
           << "chat_template=" << ChatTemplateId() << '\n'
           << "chat_template_reference_sha256=" << EncoderReferenceSha256()
           << '\n'
           << "artifact_template_sha256=" << ArtifactTemplateSha256() << '\n'
           << "state_abi="
           << (use_dspark ? DS41_STATE_ABI_DSPARK : DS41_STATE_ABI_AR) << '\n'
           << "payload_layout=ds41-rocm-continuation-v"
           << kContinuationLayoutVersion << '\n'
           << "context_tokens=" << max_context << '\n'
           << "dspark=" << (use_dspark ? 1 : 0) << '\n'
           << "max_draft_tokens=" << max_draft_tokens << '\n';
  if (use_dspark)
    identity << "support_id=" << ext::kGgufIdentityScheme << ':'
             << support_fingerprint << '\n';
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> DeepSeek41ProcessIdentity(
    const void* model, std::uint32_t max_context, bool use_dspark,
    std::uint32_t max_draft_tokens) {
  std::ostringstream identity;
  identity << "schema=gufo-process-text-continuation-v1\n"
           << "model_instance=" << model << '\n'
           << "variant=" << DS41_MODEL_SHAPE_NAME << '\n'
           << "context_tokens=" << max_context << '\n'
           << "dspark=" << (use_dspark ? 1 : 0) << '\n'
           << "max_draft_tokens=" << max_draft_tokens << '\n';
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

}  // namespace DS41_VARIANT_NAMESPACE
