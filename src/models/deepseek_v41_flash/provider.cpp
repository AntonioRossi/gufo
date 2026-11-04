#include <gufo/ext/generated/deepseek_v41_flash_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/deepseek_v41_flash/probe.hpp"
#include "src/models/deepseek_v41_flash/runner.hpp"

void deepseek_v41_flash_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_deepseek_v41_flash::kProvider_deepseek41);
  registry.AddTextProvider({
      .id = "deepseek41",
      .probe = &gufo::models::deepseek_v41_flash::ProbeDeepSeek,
      .probe_gguf = &gufo::models::deepseek_v41_flash::ProbeDeepSeekGguf,
      .create = &gufo::models::deepseek_v41_flash::CreateDeepSeekTextRunner,
      .prompt_driver = &gufo::models::deepseek_v41_flash::RunDeepSeekPrompt,
      .chat_driver = &gufo::models::deepseek_v41_flash::RunDeepSeekChat,
      .supports_expert_streaming = true,
  });
}
