#include "src/models/deepseek_v41_flash/continuation_layout.hpp"

#include <stdexcept>
#include <utility>

namespace DS41_VARIANT_NAMESPACE {
ContinuationLayout::ContinuationLayout(ContinuationGeometry geometry,
                                       std::uint32_t context)
    : geometry_(std::move(geometry)), context_(context) {
  if (!context || !geometry_.vocabulary || !geometry_.window_rows ||
      geometry_.ratios.empty() ||
      geometry_.ratios.size() != geometry_.part_bytes.size())
    throw std::invalid_argument("invalid V4.1 continuation geometry");
  const auto add = [&](ContinuationPart part, std::size_t row,
                       std::size_t state, std::uint32_t index = 0,
                       cache::Rows window = 0) {
    const auto id = static_cast<std::uint32_t>(components_.size() + 1);
    // The native ring has no slack beyond the live window. One row per chunk
    // meets window_rows + rows_per_chunk - 1 without changing native kernels.
    // Growing device rows retain their existing backing-page chunk size.
    const cache::Rows chunk = window ? 1
                              : !row ? 0
                              : part == ContinuationPart::kTokens
                                  ? 256
                                  : (65536 + row - 1) / row;
    components_.push_back({{{id},
                            kContinuationLayoutVersion,
                            window ? cache::ComponentKind::kWindowRows
                            : row  ? cache::ComponentKind::kAppendRows
                                   : cache::ComponentKind::kPrivateState,
                            row,
                            chunk,
                            state,
                            window},
                           part,
                           index});
  };
  add(ContinuationPart::kTokens, sizeof(std::int32_t), 0);
  add(ContinuationPart::kLogits, 0,
      std::size_t{geometry_.vocabulary} * sizeof(float));
  add(ContinuationPart::kMetadata, 0, kContinuationMetadataBytes);
  for (std::uint32_t layer = 0; layer < geometry_.ratios.size(); ++layer) {
    const auto& bytes = geometry_.part_bytes[layer];
    if (!bytes[0] || bytes[0] % geometry_.window_rows)
      throw std::invalid_argument("V4.1 layer has no window");
    add(ContinuationPart::kDecoderWindow, bytes[0] / geometry_.window_rows, 0,
        layer, geometry_.window_rows);
    if (!geometry_.ratios[layer]) {
      for (std::size_t part = 1; part < bytes.size(); ++part)
        if (bytes[part])
          throw std::invalid_argument("V4.1 row state has no ratio");
      continue;
    }
    if (!bytes[1] || !bytes[2] || !bytes[3] || !bytes[4] || !bytes[5] ||
        !bytes[6])
      throw std::invalid_argument("incomplete V4.1 compressed layer");
    // Ratio-one sources write compressed rows directly and do not use pools.
    if (geometry_.ratios[layer] > 1)
      add(ContinuationPart::kPoolingState, 0, bytes[5] + bytes[6], layer);
    add(ContinuationPart::kAttentionRows, bytes[1], 0, layer);
    add(ContinuationPart::kAttentionScales, bytes[2], 0, layer);
    add(ContinuationPart::kIndexerRows, bytes[3], 0, layer);
    add(ContinuationPart::kIndexerScales, bytes[4], 0, layer);
  }
  if (geometry_.draft_bytes) {
    const auto rows =
        std::size_t{geometry_.draft_stages} * geometry_.window_rows;
    if (!rows || geometry_.draft_bytes % rows)
      throw std::invalid_argument("invalid V4.1 draft ring geometry");
    for (std::uint32_t stage = 0; stage < geometry_.draft_stages; ++stage)
      add(ContinuationPart::kDraftWindow, geometry_.draft_bytes / rows, 0,
          stage, geometry_.window_rows);
  } else if (geometry_.draft_stages) {
    throw std::invalid_argument("V4.1 draft stages have no ring");
  }
}
std::vector<cache::ComponentPosition> ContinuationLayout::Positions(
    std::uint32_t position) const {
  if (position >= context_)
    throw std::invalid_argument("invalid V4.1 checkpoint position");
  std::vector<cache::ComponentPosition> result;
  for (const auto& component : components_) {
    const bool compressed =
        component.descriptor.kind == cache::ComponentKind::kAppendRows &&
        component.part != ContinuationPart::kTokens;
    result.push_back(
        {component.descriptor.id,
         compressed ? position / geometry_.ratios[component.index] : position});
  }
  return result;
}
std::size_t ContinuationLayout::PrivateBytes() const noexcept {
  std::size_t result = 0;
  for (const auto& component : components_)
    result += component.descriptor.state_bytes;
  return result;
}
std::size_t ContinuationLayout::RowBytes(std::uint32_t position) const {
  const auto positions = Positions(position);
  std::size_t result = 0;
  for (std::size_t i = 0; i < components_.size(); ++i) {
    const auto& descriptor = components_[i].descriptor;
    result += descriptor.row_bytes *
              (positions[i].valid_rows -
               cache::FirstChunkRow(descriptor, positions[i].valid_rows));
  }
  return result;
}
std::vector<cache::BackingClass> ContinuationLayout::BackingPlan(
    std::size_t ram) const {
  // Use the reference V4 backing policy, including metadata and page slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}
}  // namespace DS41_VARIANT_NAMESPACE
