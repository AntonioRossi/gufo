#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "src/models/deepseek_v41_flash/engine.hpp"
#include "src/models/deepseek_v41_flash/runner.hpp"

namespace gufo::models::deepseek_v41_flash {
namespace {

constexpr std::uint32_t kDefaultContext = 4096;

ReasoningEffort ParseReasoningEffort(std::string_view value) {
  if (value == "minimal")
    return ReasoningEffort::kMinimal;
  if (value == "low")
    return ReasoningEffort::kLow;
  if (value == "medium")
    return ReasoningEffort::kMedium;
  if (value == "high")
    return ReasoningEffort::kHigh;
  if (value == "xhigh")
    return ReasoningEffort::kXHigh;
  if (value == "max")
    return ReasoningEffort::kMax;
  throw std::invalid_argument("DeepSeek V4.1 reasoning effort is invalid");
}

ReasoningOptions PromptReasoningOptions(const ext::TextPromptOptions& options) {
  ReasoningOptions reasoning;
  if (options.reasoning_mode == "on")
    reasoning.enabled = true;
  else if (options.reasoning_mode == "off")
    reasoning.enabled = false;
  if (options.reasoning_effort != "auto") {
    reasoning.effort = ParseReasoningEffort(options.reasoning_effort);
    reasoning.enabled = true;
  }
  if (options.preserve_thinking == "on")
    reasoning.preserve_thinking = true;
  else if (options.preserve_thinking == "off")
    reasoning.preserve_thinking = false;
  return reasoning;
}

std::shared_ptr<ext::TextModelRunner> LoadRunner(
    const ext::TextPromptOptions& options, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  if (options.force_cpu)
    throw std::invalid_argument("DeepSeek V4.1 requires the ROCm backend");
  if (!options.vision_model_path.empty() || !options.image_paths.empty() ||
      options.add_vision_id)
    throw std::invalid_argument("DeepSeek V4.1 vision is not supported");
  if (!options.speculative_backend.empty() || !options.mtp_model_path.empty() ||
      !options.dflash_model_path.empty() ||
      !options.dspark_model_path.empty() || !options.draft_policy.empty() ||
      options.draft_tokens != 7 || options.min_draft_tokens != 1)
    throw std::invalid_argument(
        "DeepSeek V4.1 speculative decoding is not supported");
  ext::TextRunnerOptions runner_options;
  runner_options.model_path = options.model_path;
  runner_options.reader = std::shared_ptr<const core::GgufReader>(
      &reader, [](const core::GgufReader*) {});
  runner_options.max_context = kDefaultContext;
  runner_options.ram_cache.enabled = false;
  std::string error;
  auto runner = CreateDeepSeekTextRunner(runner_options, &error);
  if (!runner)
    throw std::runtime_error(error);
  if (load_start != std::chrono::steady_clock::time_point{}) {
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - load_start)
            .count();
    std::cerr << "Model loaded in " << milliseconds << " ms.\n";
  }
  return runner;
}

ext::ChatRequest InitialChat(const ext::TextPromptOptions& options) {
  ext::ChatRequest request;
  request.reasoning = PromptReasoningOptions(options);
  request.cache_prompt = false;
  request.cache_persist = false;
  if (!options.system_prompt.empty())
    request.messages.emplace_back(ext::ChatRole::kSystem,
                                  options.system_prompt);
  return request;
}

std::vector<ext::TextRunnerToken> ChatTokens(const ext::TextModelRunner& runner,
                                             const ext::ChatRequest& request) {
  auto tokens = runner.RenderAndTokenize(request);
  if (!tokens)
    throw std::runtime_error("DeepSeek V4.1 chat produced no tokens");
  return std::move(*tokens);
}

std::string Generate(const ext::TextPromptOptions& options,
                     const ext::TextModelRunner& runner,
                     std::span<const ext::TextRunnerToken> tokens) {
  const auto descriptor = runner.Descriptor();
  if (tokens.empty())
    throw std::invalid_argument("DeepSeek V4.1 prompt produced no tokens");
  if (tokens.size() >= descriptor.max_context ||
      options.max_tokens >= descriptor.max_context - tokens.size())
    throw std::invalid_argument(
        "DeepSeek V4.1 prompt and output exceed the CLI context");
  auto state = runner.CreateState();
  for (std::size_t offset = 0; offset < tokens.size();) {
    const auto step =
        runner.Prefill(*state, tokens, offset, tokens.size() - offset);
    if (step.consumed_tokens == 0)
      throw std::runtime_error("DeepSeek V4.1 prefill made no progress");
    offset += step.consumed_tokens;
  }
  const auto sampling_config = sampling::ResolveTextSampling(
      descriptor.sampling_preset, PromptReasoningOptions(options).enabled,
      options.sampling, options.sampling_supplied);
  sampling::SamplerState sampler(sampling_config, tokens);
  if (options.verbose)
    std::cerr << "Prompt tokens: " << tokens.size()
              << "\nMax tokens: " << options.max_tokens << '\n';
  std::string reply;
  for (std::size_t index = 0; index < options.max_tokens; ++index) {
    const auto step = runner.SelectNext(*state, sampler);
    if (step.stop)
      break;
    runner.Advance(*state, step.token);
    sampler.Accept(step.token);
    reply += step.piece;
    std::cout << step.piece << std::flush;
  }
  std::cout << '\n';
  return reply;
}

}  // namespace

int RunDeepSeekPrompt(const ext::TextPromptOptions& options,
                      const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start) {
  try {
    auto runner = LoadRunner(options, reader, load_start);
    std::vector<ext::TextRunnerToken> tokens;
    if (options.use_chat_template) {
      auto request = InitialChat(options);
      request.messages.emplace_back(ext::ChatRole::kUser, options.prompt_text);
      tokens = ChatTokens(*runner, request);
    } else {
      tokens = runner->Tokenize(options.prompt_text);
    }
    Generate(options, *runner, tokens);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "DeepSeek V4.1 prompt failed: " << error.what() << '\n';
    return 1;
  }
}

int RunDeepSeekChat(const ext::TextPromptOptions& options,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start) {
  try {
    if (!options.use_chat_template)
      throw std::invalid_argument(
          "DeepSeek V4.1 chat requires the chat template");
    auto runner = LoadRunner(options, reader, load_start);
    auto request = InitialChat(options);
    const bool thinking = runner->InitialOutputState(request) ==
                          ext::InitialOutputState::kThought;
    std::cout << "=== Gufo Interactive Chat (DeepSeek V4.1 Flash) ===\n"
              << "Type 'exit' or Ctrl+D to quit.\n\n";
    for (std::string input;;) {
      std::cout << ">>> User: " << std::flush;
      if (!std::getline(std::cin, input) || input == "exit" || input == "quit")
        break;
      if (input.empty())
        continue;
      request.messages.emplace_back(ext::ChatRole::kUser, input);
      const auto tokens = ChatTokens(*runner, request);
      std::cout << "<<< Assistant: ";
      auto reply = Generate(options, *runner, tokens);
      ext::ChatMessage response(ext::ChatRole::kAssistant, "");
      if (thinking) {
        constexpr std::string_view end = "</think>";
        const auto boundary = reply.find(end);
        response.thought = reply.substr(0, boundary);
        if (boundary != std::string::npos)
          response.content = reply.substr(boundary + end.size());
      } else {
        response.content = std::move(reply);
      }
      request.messages.push_back(std::move(response));
      std::cout << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "DeepSeek V4.1 chat failed: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace gufo::models::deepseek_v41_flash
