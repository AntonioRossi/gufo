#pragma once

#include <gufo/ext/text.hpp>

namespace gufo::models::deepseek_v41_flash {

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

}  // namespace gufo::models::deepseek_v41_flash
