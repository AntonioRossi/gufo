#ifndef GUFO_TESTS_MODELS_DEEPSEEK_V41_FLASH_IDENTITY_VARIANTS_HPP_
#define GUFO_TESTS_MODELS_DEEPSEEK_V41_FLASH_IDENTITY_VARIANTS_HPP_

#include <cstdint>
#include <string_view>
#include <vector>

namespace identity_test {

std::vector<std::uint8_t> V384Identity(std::string_view artifact_fingerprint,
                                       std::string_view support_fingerprint,
                                       std::uint32_t max_context,
                                       bool use_dspark,
                                       std::uint32_t max_draft_tokens);

std::vector<std::uint8_t> CoderIdentity(std::string_view artifact_fingerprint,
                                        std::string_view support_fingerprint,
                                        std::uint32_t max_context,
                                        bool use_dspark,
                                        std::uint32_t max_draft_tokens);

}  // namespace identity_test

#endif  // GUFO_TESTS_MODELS_DEEPSEEK_V41_FLASH_IDENTITY_VARIANTS_HPP_
