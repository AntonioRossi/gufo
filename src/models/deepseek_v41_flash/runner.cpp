#include "src/models/deepseek_v41_flash/runner.hpp"

#include <algorithm>
#include <gufo/ext/host.hpp>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "src/models/deepseek_v41_flash/engine.hpp"

namespace gufo::models::deepseek_v41_flash {
namespace {

using ext::TextRunnerToken;

struct Execution {
  std::mutex mutex;
  std::shared_ptr<Model> model;
};

class Slot final : public cache::Slot {
public:
  Slot(std::shared_ptr<Execution> execution, cache::MutationGuard& guard)
      : execution_(std::move(execution)), guard_(guard) {
    const std::lock_guard lock(execution_->mutex);
    std::string error;
    session_ = execution_->model->CreateSession(
        core::SessionMode::kAutoregressive, execution_->model->MaxContext(),
        &error);
    if (!session_)
      throw std::runtime_error("DeepSeek V4.1 session creation failed: " +
                               error);
    // This port has one session and no snapshot or prompt-cache backing.
    // The session and its long-lived buffers exist before the cache fit.
    execution_->model->FitExpertCache(0);
    const auto cache = execution_->model->ExpertCacheStatus();
    if (cache.enabled) {
      gufo::server::Logger::LogFormatted(
          gufo::server::LogLevel::kInfo, "ds41",
          "event=expert_streaming stream_experts=true expert_cache_bytes=%llu "
          "cache_bytes=%llu cache_limit_bytes=%llu",
          static_cast<unsigned long long>(cache.requested_bytes),
          static_cast<unsigned long long>(cache.allocated_bytes),
          static_cast<unsigned long long>(cache.limit_bytes));
    }
  }

  ~Slot() override {
    const std::lock_guard lock(execution_->mutex);
    session_.reset();
  }

  bool IsValid() const noexcept override { return true; }

  void SetCancellationCheck(
      const ext::TextRunnerState::CancellationCheck& check) {
    const std::lock_guard lock(execution_->mutex);
    session_->SetCancellationCheck(check);
  }

  void Invalidate() noexcept {
    const std::lock_guard lock(execution_->mutex);
    session_->SetCancellationCheck({});
    session_->Invalidate();
    guard_.AfterReset();
  }

  Session& session() const { return *session_; }
  const Execution* execution() const { return execution_.get(); }

private:
  std::shared_ptr<Execution> execution_;
  cache::MutationGuard& guard_;
  std::unique_ptr<Session> session_;
};

// V4.1 has no prompt-cache components. The adapter owns the execution slot
// and rejects all capture and restore operations.
class Adapter final : public cache::Adapter {
public:
  explicit Adapter(std::shared_ptr<Execution> execution)
      : execution_(std::move(execution)) {}

  cache::Capabilities GetCapabilities() const override { return {}; }
  std::span<const cache::ComponentDescriptor> Components() const override {
    return {};
  }
  cache::Identity CompatibilityIdentity() const override {
    const auto address = reinterpret_cast<std::uintptr_t>(execution_.get());
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&address);
    return {bytes, bytes + sizeof(address)};
  }
  std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard& guard) override {
    return std::make_unique<Slot>(execution_, guard);
  }
  std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override {
    return {};
  }
  std::vector<cache::Rows> PlanPrefill(const cache::Slot& slot,
                                       cache::Rows first,
                                       cache::Rows end) const override {
    const auto& native = RequireSlot(slot);
    const auto capacity = native.session().PrefillCapacity();
    std::vector<cache::Rows> boundaries;
    while (first < end) {
      first += std::min<cache::Rows>(capacity, end - first);
      boundaries.push_back(first);
    }
    return boundaries;
  }
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override {
    Unsupported();
  }
  cache::Completion CapturePrivate(const cache::Slot&, cache::ComponentId,
                                   std::span<std::byte>,
                                   cache::Stream&) override {
    Unsupported();
  }
  cache::Completion CopyRowsOut(const cache::Slot&, cache::ComponentId,
                                cache::Rows, cache::Rows, std::span<std::byte>,
                                cache::Stream&) override {
    Unsupported();
  }
  cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId, cache::Rows,
                               cache::Rows, std::span<const std::byte>,
                               cache::Stream&) override {
    Unsupported();
  }
  cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                std::span<const std::byte>,
                                cache::Stream&) override {
    Unsupported();
  }
  bool Validate(cache::Slot&,
                std::span<const cache::ComponentPosition>) override {
    Unsupported();
  }
  bool Invalidate(cache::Slot& slot) noexcept override {
    auto* native = dynamic_cast<Slot*>(&slot);
    if (!native || native->execution() != execution_.get())
      return false;
    native->Invalidate();
    return true;
  }

  const Slot& RequireSlot(const cache::Slot& slot) const {
    const auto* native = dynamic_cast<const Slot*>(&slot);
    if (!native || native->execution() != execution_.get())
      throw std::invalid_argument(
          "DeepSeek V4.1 slot belongs to another runner");
    return *native;
  }

private:
  [[noreturn]] static void Unsupported() {
    throw std::logic_error("DeepSeek V4.1 prompt caching is not supported");
  }
  std::shared_ptr<Execution> execution_;
};

class State final : public ext::TextRunnerState {
public:
  explicit State(Slot& slot) : slot_(slot) {}
  void SetCancellationCheck(const CancellationCheck& check) override {
    slot_.SetCancellationCheck(check);
  }
  void Invalidate() noexcept override { slot_.Invalidate(); }
  Session& session() const { return slot_.session(); }
  const Execution* execution() const { return slot_.execution(); }

private:
  Slot& slot_;
};

class Stream final : public cache::Stream {
public:
  cache::TransferResult Synchronize() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};

std::vector<TextRunnerToken> RunnerTokens(std::span<const int> tokens) {
  std::vector<TextRunnerToken> result;
  result.reserve(tokens.size());
  for (const int token : tokens) {
    if (token < 0)
      throw std::invalid_argument("DeepSeek V4.1 token ID is negative");
    result.push_back(static_cast<TextRunnerToken>(token));
  }
  return result;
}

class Runner final : public ext::HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kDeepSeek41;
  }

  explicit Runner(std::shared_ptr<Model> model)
      : execution_(std::make_shared<Execution>()) {
    execution_->model = std::move(model);
  }

  ext::TextRunnerDescriptor Descriptor() const override {
    return {.model_id = execution_->model->ModelName(),
            .state_abi = "deepseek-v41-flash-1017-ar-v1",
            .max_context = execution_->model->MaxContext(),
            .capabilities = {.incremental_prefill = true,
                             .final_token_advance_required = false,
                             .incremental_text_is_exact = true,
                             .prefix_reuse = false},
            .sampling_preset = sampling::TextModelPreset::kDeepSeekV41Flash};
  }

  ext::TextRunnerResourceClaim ResourceClaim() const override {
    ext::TextRunnerResourceClaim claim;
    claim.retained_snapshot_capacity_bytes = 0;
    claim.retained_snapshot_ceiling_bytes = ext::HostSnapshotCeilingBytes();
    claim.requires_device_runtime_lock = true;
    return claim;
  }

  std::vector<ext::TextExecutionPlan> SupportedPlans() const override {
    return {{.kind = ext::TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        execution_->model->VocabSize(),
        [model = execution_->model](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              model->DecodeToken(static_cast<int>(id)),
              model->IsStopToken(static_cast<int>(id))};
        });
  }

  std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(execution_->model->VocabSize());
  }

  std::vector<TextRunnerToken> Tokenize(std::string_view text) const override {
    return RunnerTokens(execution_->model->Tokenize(text, true));
  }

  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ext::ChatRequest& request) const override {
    const auto input = ConvertChat(request);
    auto tokens = RunnerTokens(execution_->model->EncodeChat(
        input.messages, input.tools, input.options));
    if (tokens.empty())
      return std::nullopt;
    return tokens;
  }

  std::optional<std::string> RenderPrompt(
      const ext::ChatRequest& request) const override {
    const auto input = ConvertChat(request);
    return RenderChat(input.messages, input.tools, input.options);
  }

  ext::InitialOutputState InitialOutputState(
      const ext::ChatRequest& request) const override {
    return ResolveDeepSeekChatOptions(request.reasoning).enable_thinking
               ? ext::InitialOutputState::kThought
               : ext::InitialOutputState::kText;
  }

  std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const auto token : tokens)
      text += execution_->model->DecodeToken(EngineToken(token));
    return text;
  }

  std::unique_ptr<ext::ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger&, std::size_t) const override {
    auto resources = std::make_unique<ext::ComponentCacheResources>();
    resources->adapter = std::make_unique<Adapter>(execution_);
    resources->stream = [] { return std::make_unique<Stream>(); };
    return resources;
  }

  std::unique_ptr<ext::TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    const auto& native = dynamic_cast<Adapter&>(adapter).RequireSlot(slot);
    if (native.execution() != execution_.get())
      throw std::invalid_argument(
          "DeepSeek V4.1 slot belongs to another runner");
    return std::make_unique<State>(dynamic_cast<Slot&>(slot));
  }

  void ReconcileComponentState(ext::TextRunnerState& state,
                               std::size_t position) const override {
    const std::lock_guard lock(execution_->mutex);
    if (position !=
        static_cast<std::size_t>(RequireState(state).session().Position()))
      throw std::invalid_argument("DeepSeek V4.1 cache position is invalid");
  }

  ext::TextPrefillStep Prefill(ext::TextRunnerState& state,
                               std::span<const TextRunnerToken> prompt,
                               std::size_t offset,
                               std::size_t max_input_tokens) const override {
    const std::lock_guard lock(execution_->mutex);
    auto& session = RequireState(state).session();
    if (offset != static_cast<std::size_t>(session.Position()) ||
        offset >= prompt.size() || max_input_tokens == 0 ||
        prompt.size() >= execution_->model->MaxContext())
      throw std::invalid_argument("DeepSeek V4.1 prefill range is invalid");
    const auto consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, session.PrefillCapacity()});
    std::vector<int> prefix;
    prefix.reserve(offset + consumed);
    for (const auto token : prompt.first(offset + consumed))
      prefix.push_back(EngineToken(token));
    std::string error;
    if (!session.Sync(prefix, &error, prompt.size()))
      throw std::runtime_error("DeepSeek V4.1 prefill failed: " + error);
    return {.consumed_tokens = consumed,
            .decode_ready = offset + consumed == prompt.size()};
  }

  ext::TextDecodeSelection SelectNext(
      ext::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    const std::lock_guard lock(execution_->mutex);
    auto& native = RequireState(state);
    if (native.session().Position() >=
        static_cast<int>(execution_->model->MaxContext()))
      return {.stop = true, .piece = {}, .logprobs = std::nullopt};
    std::string error;
    const auto logits = native.session().CopyLogits(&error);
    if (logits.empty())
      throw std::runtime_error("DeepSeek V4.1 logits copy failed: " + error);
    const auto token = sampler.Sample(logits);
    const bool stop = native.stop_at_eos() &&
                      execution_->model->IsStopToken(EngineToken(token));
    return {.stop = stop,
            .token = token,
            .piece = stop ? std::string{}
                          : execution_->model->DecodeToken(EngineToken(token)),
            .logprobs = sampling::CollectLogprobs(logits, token,
                                                  sampler.config().logprobs)};
  }

  void Advance(ext::TextRunnerState& state,
               TextRunnerToken token) const override {
    const std::lock_guard lock(execution_->mutex);
    std::string error;
    if (!RequireState(state).session().Evaluate(EngineToken(token), &error))
      throw std::runtime_error("DeepSeek V4.1 decode failed: " + error);
  }

  std::size_t CheckpointPosition(
      const ext::TextRunnerState& state) const override {
    const std::lock_guard lock(execution_->mutex);
    return static_cast<std::size_t>(RequireState(state).session().Position());
  }

private:
  int EngineToken(TextRunnerToken token) const {
    if (token >= VocabularySize())
      throw std::invalid_argument(
          "DeepSeek V4.1 token ID is outside the vocabulary");
    return static_cast<int>(token);
  }

  const State& RequireState(const ext::TextRunnerState& state) const {
    const auto* native = dynamic_cast<const State*>(&state);
    if (!native || native->execution() != execution_.get())
      throw std::invalid_argument(
          "DeepSeek V4.1 state belongs to another runner");
    return *native;
  }

  std::shared_ptr<Execution> execution_;
};

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateDeepSeekTextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  try {
    if (options.session_count != 1)
      throw std::invalid_argument("DeepSeek V4.1 requires one session");
    const ext::TextSpeculativeConfig speculative_defaults;
    if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled ||
        !options.speculative.draft_model_path.empty() ||
        options.speculative.max_draft_tokens !=
            speculative_defaults.max_draft_tokens ||
        options.speculative.min_draft_tokens !=
            speculative_defaults.min_draft_tokens ||
        options.speculative.dflash_policy != speculative_defaults.dflash_policy)
      throw std::invalid_argument(
          "DeepSeek V4.1 speculative decoding is not supported");
    if (!options.vision_model_path.empty())
      throw std::invalid_argument("DeepSeek V4.1 vision is not supported");
    if (!options.disk_cache.directory.empty() ||
        (options.ram_cache.enabled && options.ram_cache.capacity_bytes != 0))
      throw std::invalid_argument(
          "DeepSeek V4.1 prompt caching is not supported");
    if (options.enable_yarn)
      throw std::invalid_argument(
          "--enable-yarn is not supported by DeepSeek V4.1");
    auto reader = options.reader;
    std::string detail;
    if (!reader)
      reader = core::GgufReader::OpenFile(options.model_path, &detail);
    if (!reader)
      throw std::runtime_error("Cannot open DeepSeek V4.1 GGUF: " + detail);
    if (!ValidateGgufTemplate(*reader, &detail))
      throw std::invalid_argument(detail);
    std::uint64_t context = options.max_context;
    if (context == 0)
      context =
          reader->GetMetadataUint64("deepseek41.context_length").value_or(0);
    if (context < 2 ||
        context > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("DeepSeek V4.1 needs a valid context length");
    ModelOptions model_options;
    model_options.max_context = static_cast<std::uint32_t>(context);
    model_options.stream_experts = options.stream_experts;
    model_options.expert_cache_bytes = options.expert_cache_bytes;
    auto model = Model::Load(options.model_path, model_options, &detail);
    if (!model)
      throw std::runtime_error("Cannot load DeepSeek V4.1: " + detail);
    return std::make_shared<Runner>(std::move(model));
  } catch (const std::exception& exception) {
    if (error)
      *error = exception.what();
    return nullptr;
  }
}

}  // namespace gufo::models::deepseek_v41_flash
