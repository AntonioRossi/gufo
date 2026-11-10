#include "src/models/deepseek_v41_flash/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <gufo/ext/snapshot_copy.hpp>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/models/deepseek_v41_flash/runtime/model_data_internal.h"
#include "src/models/deepseek_v41_flash/runtime/v41_dspark.h"

namespace DS41_VARIANT_NAMESPACE {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
struct Metadata {
  std::uint32_t version{kContinuationLayoutVersion};
  std::uint32_t position{}, decoder_from{}, draft_start{}, draft_end{};
  ds41_session_request_state request;
};
static_assert(sizeof(float) == 4 && sizeof(Metadata) == 92);
static_assert(offsetof(Metadata, request) == 20);
static_assert(sizeof(ds41_session_request_state) == 72);
static_assert(offsetof(ds41_session_request_state, cycle_ms) == 20);
static_assert(offsetof(ds41_session_request_state, draft_ms) == 44);
static_assert(offsetof(ds41_session_request_state, plain) == 52);
static_assert(offsetof(ds41_session_request_state, lookup_resume) == 60);
static_assert(offsetof(ds41_session_request_state, next) == 68);
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
static_assert(std::is_trivially_copyable_v<Metadata>);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(Coverage ranges, std::size_t first, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = first;
  for (const auto& [begin, last] : ranges) {
    if (begin < first || begin > frontier || last > end)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}
ContinuationGeometry ModelGeometry(ds41_session* session) {
  ContinuationGeometry geometry;
  geometry.vocabulary = DS41_N_VOCAB;
  geometry.window_rows = DS41_N_SWA;
  auto* decoder = ds41_session_decoder(session);
  for (std::uint32_t layer = 0; layer < DS41_N_LAYER; ++layer) {
    geometry.ratios.push_back(ds41_layer_state_ratio(layer));
    std::array<std::size_t, 7> parts{};
    for (int part = DS41_STATE_WINDOW; part <= DS41_STATE_POOL_SCORE; ++part) {
      std::uint64_t bytes = 0;
      if (ds41_decoder_state_part(decoder, layer, part, &bytes))
        parts[part] = bytes;
    }
    geometry.part_bytes.push_back(parts);
  }
  std::uint64_t bytes = 0;
  if (ds41_session_dspark_ring(session, &bytes)) {
    geometry.draft_stages = DS41_DSPARK_STAGES;
    geometry.draft_bytes = bytes;
  }
  return geometry;
}
}  // namespace

class ContinuationAdapter::State final : public cache::Slot {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    AttachHooks();
  }
  void AttachHooks() {
    ds41_session_set_write_hook(
        native(),
        [](void* data, std::uint32_t first, std::uint32_t end) {
          static_cast<State*>(data)->BeforeWrite(first, end);
        },
        this, [](void* data) { static_cast<State*>(data)->BeforeReset(); },
        [](void* data) { static_cast<State*>(data)->BeforeRestore(); });
  }
  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };
  ~State() override {
    for (const auto& job : jobs)
      (void)job->Wait();
    (void)hipStreamSynchronize(nullptr);
    Release();
    ds41_session_set_write_hook(native(), nullptr, nullptr);
    const std::lock_guard lock(adapter.pool_mutex_);
    adapter.free_sessions_.push_back(std::move(session));
  }
  [[nodiscard]] ds41_session* native() const { return session->session_; }
  bool IsValid() const noexcept override {
    return !restoring && !failed && ds41_session_coherent(native());
  }
  void RequireIdle() const {
    if (restoring || failed || pending || !ds41_session_coherent(native()))
      throw std::logic_error("DeepSeek slot is not ready for execution");
  }
  void FinishExecution() {
    const std::lock_guard lock(jobs_mutex);
    if (!execution_fenced) {
      // Target commits and DSpark ring injection use the native default
      // stream. Cache transfers use nonblocking streams and need this fence.
      Check(hipStreamSynchronize(nullptr));
      execution_fenced = true;
    }
  }
  void BeforeWrite(std::uint32_t first, std::uint32_t end) {
    RequireIdle();
    for (const auto& component : adapter.layout_.Components()) {
      if (component.descriptor.kind == cache::ComponentKind::kWindowRows) {
        const auto capacity = component.descriptor.window_rows;
        // Guard both displaced absolute rows and writes to the named range.
        // A speculative verification can write more rows than it commits.
        const auto displaced_first = first > capacity ? first - capacity : 0;
        const auto displaced_end = end > capacity ? end - capacity : 0;
        if (displaced_first < displaced_end)
          guard.BeforeOverwrite(component.descriptor.id, displaced_first,
                                displaced_end);
        guard.BeforeOverwrite(component.descriptor.id, first, end);
        continue;
      }
      if (component.descriptor.kind != cache::ComponentKind::kAppendRows)
        continue;
      const auto ratio =
          component.part == ContinuationPart::kTokens
              ? 1
              : adapter.layout_.Geometry().ratios[component.index];
      if (first / ratio < end / ratio)
        guard.BeforeOverwrite(component.descriptor.id, first / ratio,
                              end / ratio);
    }
    const std::lock_guard lock(jobs_mutex);
    execution_fenced = false;
  }
  void BeforeReset() {
    RequireIdle();
    GuardAll();
    const std::lock_guard lock(jobs_mutex);
    execution_fenced = false;
  }
  void BeforeRestore() {
    throw std::logic_error("DeepSeek slots restore through the adapter");
  }
  [[nodiscard]] Rows Capacity(const ContinuationComponent& component) const {
    return component.part == ContinuationPart::kTokens ||
                   component.descriptor.kind ==
                       cache::ComponentKind::kWindowRows
               ? adapter.context_
               : adapter.context_ /
                     adapter.layout_.Geometry().ratios[component.index];
  }
  void GuardAll() {
    for (const auto& component : adapter.layout_.Components())
      if (cache::HasRows(component.descriptor.kind))
        guard.BeforeOverwrite(component.descriptor.id, 0, Capacity(component));
  }
  void Release() noexcept {
    for (const auto& component : adapter.layout_.Components())
      if (cache::HasRows(component.descriptor.kind))
        guard.BeforeRelease(component.descriptor.id, 0, Capacity(component));
  }
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  bool restoring{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  bool execution_fenced{false};
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  std::array<std::byte, kContinuationMetadataBytes> metadata{};
};

struct ContinuationAdapter::Region {
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  std::vector<Piece> pieces;
};

ContinuationAdapter::ContinuationAdapter(
    std::shared_ptr<Model> model, core::SessionMode mode, std::uint32_t context,
    cache::Identity identity, std::vector<std::unique_ptr<Session>> sessions)
    : model_(std::move(model)),
      mode_(mode),
      context_(context),
      layout_(
          [&] {
            if (sessions.empty() || !sessions.front())
              throw std::invalid_argument(
                  "V4.1 adapter requires allocated sessions");
            return ModelGeometry(sessions.front()->session_);
          }(),
          context),
      identity_(std::move(identity)),
      free_sessions_(std::move(sessions)) {
  if ((mode != core::SessionMode::kAutoregressive &&
       mode != core::SessionMode::kSpeculative) ||
      identity_.empty() || !model_ || context > model_->MaxContext() ||
      (mode == core::SessionMode::kSpeculative && !model_->HasDspark()))
    throw std::invalid_argument("invalid DeepSeek adapter configuration");
  for (const auto& session : free_sessions_)
    if (!session || session->model_ != model_ ||
        session->ContextSize() != static_cast<int>(context_) ||
        session->DsparkEnabled() != (mode_ == core::SessionMode::kSpeculative))
      throw std::invalid_argument(
          "V4.1 adapter session differs from its model or mode");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(kContinuationLayoutVersion);
  put(context);
  put(mode == core::SessionMode::kSpeculative);
  const std::string_view abi = mode == core::SessionMode::kSpeculative
                                   ? DS41_STATE_ABI_DSPARK
                                   : DS41_STATE_ABI_AR;
  identity_.insert(identity_.end(), abi.begin(), abi.end());
  for (const auto& component : layout_.Components())
    descriptors_.push_back(component.descriptor);
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another DeepSeek adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.RequireIdle();
  return *state.session;
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  const std::lock_guard lock(pool_mutex_);
  if (free_sessions_.empty())
    throw std::runtime_error("V4.1 has no free allocated session");
  auto session = std::move(free_sessions_.back());
  free_sessions_.pop_back();
  session->ResetForReuse();
  if (!ds41_session_coherent(session->session_))
    throw std::runtime_error("V4.1 session reset failed");
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid DeepSeek slot");
  return layout_.Positions(
      static_cast<std::uint32_t>(ds41_session_pos(state.native())));
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid DeepSeek prefill plan");
  std::vector<Rows> result;
  const Rows capacity = state.session->PrefillCapacity();
  while (first < end) {
    first += std::min(end - first, capacity);
    result.push_back(first);
  }
  return result;
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid DeepSeek component ID");
  return layout_.Components()[id.value - 1];
}
void* ContinuationAdapter::RowData(
    State& state, const ContinuationComponent& component) const {
  if (component.part == ContinuationPart::kTokens)
    return const_cast<int*>(ds41_session_tokens(state.native()));
  if (component.part == ContinuationPart::kDraftWindow) {
    std::uint64_t bytes = 0;
    auto* data = ds41_session_dspark_ring(state.native(), &bytes);
    if (!data || bytes != layout_.Geometry().draft_bytes)
      throw std::logic_error("V4.1 draft geometry changed");
    const auto stage_bytes =
        component.descriptor.row_bytes * layout_.Geometry().window_rows;
    return static_cast<std::byte*>(data) + component.index * stage_bytes;
  }
  int part;
  switch (component.part) {
    case ContinuationPart::kDecoderWindow:
      part = DS41_STATE_WINDOW;
      break;
    case ContinuationPart::kAttentionRows:
      part = DS41_STATE_COMP;
      break;
    case ContinuationPart::kAttentionScales:
      part = DS41_STATE_COMP_SCALE;
      break;
    case ContinuationPart::kIndexerRows:
      part = DS41_STATE_INDEX_K;
      break;
    case ContinuationPart::kIndexerScales:
      part = DS41_STATE_INDEX_SCALE;
      break;
    default:
      throw std::invalid_argument("component has no rows");
  }
  std::uint64_t bytes = 0;
  auto* data = ds41_decoder_state_part(ds41_session_decoder(state.native()),
                                       component.index, part, &bytes);
  const auto expected = component.descriptor.row_bytes *
                        (component.part == ContinuationPart::kDecoderWindow
                             ? layout_.Geometry().window_rows
                             : 1);
  if (!data || bytes != expected)
    throw std::logic_error("V4.1 row geometry changed");
  return data;
}

ContinuationAdapter::Region ContinuationAdapter::RowRegion(
    State& state, const ContinuationComponent& component, Rows first,
    Rows end) const {
  Region result;
  if (first == end)
    return result;
  const auto& d = component.descriptor;
  auto* data = static_cast<std::byte*>(RowData(state, component));
  std::size_t offset = 0;
  while (first < end) {
    const auto physical = d.window_rows ? first % d.window_rows : first;
    const auto rows = d.window_rows
                          ? std::min(end - first, d.window_rows - physical)
                          : end - first;
    const auto bytes = rows * d.row_bytes;
    result.pieces.push_back({data + physical * d.row_bytes, offset, bytes,
                             component.part == ContinuationPart::kTokens});
    first += rows;
    offset += bytes;
  }
  return result;
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  state.RequireIdle();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete DeepSeek restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate DeepSeek frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  if (ordered.front().valid_rows == 0 || ordered.front().valid_rows >= context_)
    throw std::invalid_argument("DeepSeek frontier exceeds context");
  const auto target = static_cast<std::uint32_t>(ordered.front().valid_rows);
  if (ordered != layout_.Positions(target))
    throw std::invalid_argument("inconsistent DeepSeek component frontiers");
  // Reserve first; any refusal leaves execution valid. Guard all ranges while
  // the original state is still readable.
  std::vector<Coverage> coverage(descriptors_.size());
  state.FinishExecution();
  state.GuardAll();
  if (!ds41_session_begin_restore(state.native(), target))
    throw std::runtime_error("DeepSeek restore could not reserve its rows");
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata.fill(std::byte{});
  state.restoring = true;
  try {
    // A shorter restore must not expose rows from the previous ring. Clear
    // unused slots as well, so native snapshots keep their reset-byte form.
    for (const auto& component : layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kWindowRows)
        Check(hipMemsetAsync(
            RowData(state, component), 0,
            component.descriptor.window_rows * component.descriptor.row_bytes,
            nullptr));
    Check(hipStreamSynchronize(nullptr));
  } catch (...) {
    state.failed = true;
    throw;
  }
}

cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

// Private components retain the compressor frontier and continuation metadata.
ContinuationAdapter::Region ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& component,
    std::uint32_t position) const {
  (void)position;
  Region result;
  std::size_t offset = 0;
  const auto add = [&](void* pointer, std::size_t bytes, bool host = false) {
    if (bytes) {
      if (!pointer)
        throw std::logic_error("missing V4.1 private tensor");
      result.pieces.push_back({pointer, offset, bytes, host});
    }
    offset += bytes;
  };
  switch (component.part) {
    case ContinuationPart::kLogits:
      add(ds41_session_logits(state.native()), component.descriptor.state_bytes,
          true);
      break;
    case ContinuationPart::kMetadata:
      add(state.metadata.data(), state.metadata.size(), true);
      break;
    case ContinuationPart::kPoolingState:
      for (int part : {DS41_STATE_POOL_KV, DS41_STATE_POOL_SCORE}) {
        std::uint64_t bytes = 0;
        auto* data =
            ds41_decoder_state_part(ds41_session_decoder(state.native()),
                                    component.index, part, &bytes);
        if (data)
          add(data, bytes);
      }
      break;
    default:
      throw std::invalid_argument("component is not private state");
  }
  if (offset != component.descriptor.state_bytes)
    throw std::logic_error("V4.1 private geometry changed");
  return result;
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect DeepSeek private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid())
    throw std::logic_error("cannot capture an invalid DeepSeek frontier");
  state.FinishExecution();
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid DeepSeek private piece");
  auto* native = state.native();
  const auto position = static_cast<std::uint32_t>(ds41_session_pos(native));
  if (!ds41_session_capturable(native))
    throw std::logic_error("DeepSeek rows disagree with the frontier");
  std::array<std::byte, kContinuationMetadataBytes> host_metadata{};
  auto region = [&] {
    if (component.part == ContinuationPart::kMetadata) {
      Metadata metadata{.position = position,
                        .decoder_from = ds41_session_decoder_from(native),
                        .request = ds41_session_request(native)};
      if (mode_ == core::SessionMode::kSpeculative &&
          (ds41_session_dspark_range(native, &metadata.draft_start,
                                     &metadata.draft_end) != 0 ||
           metadata.draft_end != position))
        throw std::runtime_error("DSpark prefix lacks complete support state");
      std::memcpy(host_metadata.data(), &metadata, sizeof(metadata));
      return Region{{{host_metadata.data(), 0, host_metadata.size(), true}}};
    }
    return PrivateRegion(state, component, position);
  }();
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        // Pieces overwrite their ranges. Clear only uncovered bytes: clearing
        // the whole recycled copy on the host dominates capture time.
        std::vector<std::pair<std::size_t, std::size_t>> covered;
        covered.reserve(region.pieces.size());
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first < end)
            covered.emplace_back(first - offset, end - offset);
        }
        std::ranges::sort(covered);
        std::size_t cleared = 0;
        for (const auto& [first, end] : covered) {
          if (first > cleared)
            std::fill(bytes.begin() + cleared, bytes.begin() + first,
                      std::byte{});
          cleared = std::max(cleared, end);
        }
        std::fill(bytes.begin() + std::min(cleared, bytes.size()), bytes.end(),
                  std::byte{});
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first >= end)
            continue;
          auto* source =
              static_cast<std::byte*>(piece.pointer) + first - piece.offset;
          auto* destination = bytes.data() + first - offset;
          if (piece.host)
            std::memcpy(destination, source, end - first);
          else
            Check(hip::CopySnapshotToHost(destination, source, end - first,
                                          transfer.Native()));
        }
      });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("DeepSeek load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect DeepSeek private buffer size");
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("DeepSeek load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid DeepSeek private piece");
    const auto region = PrivateRegion(
        state, component,
        static_cast<std::uint32_t>(state.expected.front().valid_rows));
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : region.pieces) {
            const auto first = std::max(offset, piece.offset);
            const auto end =
                std::min(offset + bytes.size(), piece.offset + piece.bytes);
            if (first >= end)
              continue;
            auto* destination =
                static_cast<std::byte*>(piece.pointer) + first - piece.offset;
            const auto* source = bytes.data() + first - offset;
            if (piece.host)
              std::memcpy(destination, source, end - first);
            else
              Check(hip::CopySnapshotToDevice(destination, source, end - first,
                                              transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
// Copy FP4 codes and scales separately. Window ranges use absolute row numbers
// and split at the native ring boundary. No mutable ring view is borrowed.
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid DeepSeek slot");
  state.FinishExecution();
  const auto positions = Positions(slot);
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > positions[id.value - 1].valid_rows ||
      first < cache::FirstChunkRow(component.descriptor,
                                   positions[id.value - 1].valid_rows) ||
      bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid DeepSeek row capture");
  const auto region = RowRegion(state, component, first, end);
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        for (const auto& piece : region.pieces) {
          auto* destination = bytes.data() + piece.offset;
          if (piece.host)
            std::memcpy(destination, piece.pointer, piece.bytes);
          else
            Check(hip::CopySnapshotToHost(destination, piece.pointer,
                                          piece.bytes, transfer.Native()));
        }
      });
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("DeepSeek load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        first < cache::FirstChunkRow(component.descriptor,
                                     state.expected[id.value - 1].valid_rows) ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid DeepSeek row restore");
    const auto region = RowRegion(state, component, first, end);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : region.pieces) {
            const auto* source = bytes.data() + piece.offset;
            if (piece.host)
              std::memcpy(piece.pointer, source, piece.bytes);
            else
              Check(hip::CopySnapshotToDevice(piece.pointer, source,
                                              piece.bytes, transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("DeepSeek validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    const auto first = descriptor.row_bytes *
                       cache::FirstChunkRow(descriptor, expected.valid_rows);
    if (!Covered(state.coverage[i], first, total))
      return refuse();
  }
  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  const auto target = state.expected.front().valid_rows;
  const bool speculative = mode_ == core::SessionMode::kSpeculative;
  if (metadata.version != kContinuationLayoutVersion ||
      metadata.position != target || metadata.decoder_from >= target ||
      (speculative && (metadata.draft_start > metadata.draft_end ||
                       metadata.draft_end != target)) ||
      (!speculative && (metadata.draft_start || metadata.draft_end ||
                        metadata.request.next != -1)))
    return refuse();
  if (ds41_session_finish_restore(state.native(), metadata.decoder_from) != 0)
    return refuse();
  // The native finish operation clears the range, not the transferred ring
  // bytes. Adopt the saved range after that reset.
  if (speculative &&
      ds41_session_dspark_set_range(state.native(), metadata.draft_start,
                                    metadata.draft_end) != 0)
    return refuse();
  if (!ds41_session_restore_request(state.native(), metadata.request))
    return refuse();
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& state = As(slot);
    if (state.pending)
      return false;
    state.FinishExecution();
    state.Release();
    state.failed = true;
    ds41_session_set_write_hook(state.native(), nullptr, nullptr);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.execution_fenced = false;
    }
    state.session->ResetForReuse();
    state.AttachHooks();
    // Reset writes can remain on the default stream after hipMemset returns.
    // The disabled hook cannot mark that work for the restore copy streams.
    state.FinishExecution();
    if (!ds41_session_coherent(state.native())) {
      state.failed = true;
      return false;
    }
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}
}  // namespace DS41_VARIANT_NAMESPACE
