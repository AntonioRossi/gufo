#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_CONTINUATION_LAYOUT_HPP_

#include <array>
#include <gufo/ext/adapter.hpp>
#include <span>
#include <vector>

#include "src/models/deepseek_v41_flash/runtime/variant.h"

namespace DS41_VARIANT_NAMESPACE {

enum class ContinuationPart {
  kTokens,
  kLogits,
  kMetadata,
  kLayerState,
  kAttentionRows,
  kAttentionScales,
  kIndexerRows,
  kIndexerScales,
  kDraftWindow,
};
struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t index{};
};
struct ContinuationGeometry {
  std::vector<std::uint32_t> ratios;
  // Native ds41_state_part order. Zero denotes a part absent from the layer.
  std::vector<std::array<std::size_t, 7>> part_bytes;
  std::uint32_t vocabulary{};
  std::size_t draft_bytes{};
};
inline constexpr std::size_t kContinuationMetadataBytes = 128;
class ContinuationLayout {
public:
  ContinuationLayout(ContinuationGeometry, std::uint32_t context);
  std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  const ContinuationGeometry& Geometry() const { return geometry_; }
  std::vector<cache::ComponentPosition> Positions(std::uint32_t position) const;
  std::size_t PrivateBytes() const noexcept;
  std::size_t RowBytes(std::uint32_t position) const;
  std::vector<cache::BackingClass> BackingPlan(std::size_t ram_bytes) const;

private:
  ContinuationGeometry geometry_;
  std::uint32_t context_;
  std::vector<ContinuationComponent> components_;
};
}  // namespace DS41_VARIANT_NAMESPACE
#endif
