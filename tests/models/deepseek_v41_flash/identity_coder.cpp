#define DS41_VARIANT_CODER 1
#define DS41_VARIANT_NAMESPACE gufo::models::deepseek_v41_flash::coder
#include "src/models/deepseek_v41_flash/continuation_identity.cpp"
#include "tests/models/deepseek_v41_flash/identity_variants.hpp"

namespace identity_test {

std::vector<std::uint8_t> CoderIdentity(std::string_view artifact_fingerprint,
                                        std::string_view support_fingerprint,
                                        std::uint32_t max_context,
                                        bool use_dspark,
                                        std::uint32_t max_draft_tokens) {
  return gufo::models::deepseek_v41_flash::coder::
      DeepSeek41CompatibilityIdentity(artifact_fingerprint, support_fingerprint,
                                      max_context, use_dspark,
                                      max_draft_tokens);
}

}  // namespace identity_test
