#include <gufo/ext/generated/deepseek_v41_flash_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/deepseek_v41_flash/probe.hpp"
#include "src/models/deepseek_v41_flash/variant_api.hpp"

void deepseek_v41_flash_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_deepseek_v41_flash::kProvider_deepseek41);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "deepseek41",
      .probe = &gufo::models::deepseek_v41_flash::ProbeDeepSeekV384,
      .probe_gguf = &gufo::models::deepseek_v41_flash::ProbeDeepSeekV384Gguf,
      .create =
          &gufo::models::deepseek_v41_flash::v384::CreateDeepSeekTextRunner,
      .bench_driver =
          &gufo::models::deepseek_v41_flash::v384::RunDeepSeekBenchmark,
      .prompt_driver =
          &gufo::models::deepseek_v41_flash::v384::RunDeepSeekPrompt,
      .chat_driver = &gufo::models::deepseek_v41_flash::v384::RunDeepSeekChat,
      .supports_expert_streaming = true,
  });
#endif
}

void deepseek_v41_flash_coder_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_deepseek_v41_flash::
                           kProvider_deepseek41_coder);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "deepseek41-coder",
      .probe = &gufo::models::deepseek_v41_flash::ProbeDeepSeekCoder,
      .probe_gguf = &gufo::models::deepseek_v41_flash::ProbeDeepSeekCoderGguf,
      .create =
          &gufo::models::deepseek_v41_flash::coder::CreateDeepSeekTextRunner,
      .bench_driver =
          &gufo::models::deepseek_v41_flash::coder::RunDeepSeekBenchmark,
      .prompt_driver =
          &gufo::models::deepseek_v41_flash::coder::RunDeepSeekPrompt,
      .chat_driver = &gufo::models::deepseek_v41_flash::coder::RunDeepSeekChat,
      .supports_expert_streaming = true,
  });
#endif
}
