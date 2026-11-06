#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_VARIANT_API_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_VARIANT_API_HPP_

#include <chrono>
#include <gufo/ext/text.hpp>
#include <memory>

namespace gufo::models::deepseek_v41_flash::v384 {

std::shared_ptr<ext::TextModelRunner> CreateDeepSeekTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);
int RunDeepSeekBenchmark(const ext::TextBenchOptions& options,
                         const std::shared_ptr<const core::GgufReader>& reader,
                         std::chrono::steady_clock::time_point load_start);
int RunDeepSeekPrompt(const ext::TextPromptOptions& options,
                      const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start);
int RunDeepSeekChat(const ext::TextPromptOptions& options,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::models::deepseek_v41_flash::v384

namespace gufo::models::deepseek_v41_flash::coder {

std::shared_ptr<ext::TextModelRunner> CreateDeepSeekTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);
int RunDeepSeekBenchmark(const ext::TextBenchOptions& options,
                         const std::shared_ptr<const core::GgufReader>& reader,
                         std::chrono::steady_clock::time_point load_start);
int RunDeepSeekPrompt(const ext::TextPromptOptions& options,
                      const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start);
int RunDeepSeekChat(const ext::TextPromptOptions& options,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::models::deepseek_v41_flash::coder

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_VARIANT_API_HPP_
