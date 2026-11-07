#include "src/models/deepseek_v41_flash/continuation_layout.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace ds41 = DS41_VARIANT_NAMESPACE;
namespace cache = gufo::cache;

namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// V4.1 state geometry from ds41_decoder_state_part: four shared KV sources,
// separate FP4 codes/scales, and one sliding window in every layer.
ds41::ContinuationGeometry Production(bool speculative) {
  ds41::ContinuationGeometry geometry;
  geometry.vocabulary = 129280;
  geometry.ratios.resize(40);
  geometry.part_bytes.resize(40);
  for (auto& bytes : geometry.part_bytes)
    bytes[0] = 128 * 512 * sizeof(float);
  for (const unsigned layer : {2U, 8U, 14U, 20U}) {
    const unsigned ratio = layer == 20 ? 1 : 2;
    geometry.ratios[layer] = ratio;
    geometry.part_bytes[layer] = {
        128 * 512 * sizeof(float),  256, 32, 64, 4, ratio * 512 * sizeof(float),
        ratio * 512 * sizeof(float)};
  }
  if (speculative)
    geometry.draft_bytes = 3 * 128 * 512 * sizeof(float);
  return geometry;
}

std::size_t Count(const ds41::ContinuationLayout& layout,
                  ds41::ContinuationPart part) {
  return std::ranges::count(layout.Components(), part,
                            &ds41::ContinuationComponent::part);
}

void TestInventory() {
  const ds41::ContinuationLayout layout(Production(false), 131072);
  Require(Count(layout, ds41::ContinuationPart::kLayerState) == 40,
          "every layer keeps a sliding window");
  for (const auto part : {ds41::ContinuationPart::kAttentionRows,
                          ds41::ContinuationPart::kAttentionScales,
                          ds41::ContinuationPart::kIndexerRows,
                          ds41::ContinuationPart::kIndexerScales})
    Require(Count(layout, part) == 4, "only KV sources own compressed rows");
  Require(Count(layout, ds41::ContinuationPart::kDraftWindow) == 0,
          "AR has no draft window");
  std::uint32_t id = 0;
  for (const auto& component : layout.Components())
    Require(component.descriptor.id.value == ++id, "IDs are dense from one");
  const ds41::ContinuationLayout speculative(Production(true), 131072);
  Require(Count(speculative, ds41::ContinuationPart::kDraftWindow) == 1,
          "the native ring contains all three draft stages");
  Require(speculative.PrivateBytes() - layout.PrivateBytes() ==
              3 * 128 * 512 * sizeof(float),
          "draft ring bytes");
}

void TestPrivateBytes() {
  const ds41::ContinuationLayout layout(Production(false), 4096);
  const auto expected =
      129280 * sizeof(float) + ds41::kContinuationMetadataBytes +
      40 * 128 * 512 * sizeof(float) + (3 * 2 + 1) * 2 * 512 * sizeof(float);
  Require(layout.PrivateBytes() == expected,
          "logits, metadata, windows, and compressor frontiers are private");
}

void TestPositions() {
  const ds41::ContinuationLayout layout(Production(true), 4096);
  const auto positions = layout.Positions(1031);
  Require(positions.size() == layout.Components().size(),
          "one position per component");
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const auto& component = layout.Components()[i];
    const bool compressed = component.descriptor.row_bytes &&
                            component.part != ds41::ContinuationPart::kTokens;
    const auto expected =
        compressed ? 1031 / layout.Geometry().ratios[component.index] : 1031;
    Require(positions[i].valid_rows == expected,
            "codes and scales count only completed rows");
  }
  bool rejected = false;
  try {
    (void)layout.Positions(4097);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "a frontier beyond context is rejected");
}

void TestRowsAndBacking() {
  const ds41::ContinuationLayout layout(Production(false), 131072);
  const std::size_t tokens = 8192;
  Require(layout.RowBytes(tokens) ==
              tokens * sizeof(std::int32_t) +
                  (3 * (tokens / 2) + tokens) * (256 + 32 + 64 + 4),
          "FP4 codes and scales have distinct row widths");
  for (const auto& component : layout.Components()) {
    const auto& d = component.descriptor;
    if (d.kind != cache::ComponentKind::kAppendRows)
      Require(!d.rows_per_chunk && !d.row_bytes, "private state has no rows");
    else if (component.part != ds41::ContinuationPart::kTokens)
      Require(d.rows_per_chunk * d.row_bytes >= 65536,
              "a device-row chunk fills a backing page");
  }
  Require(layout.BackingPlan(1 << 20).empty(), "tiny budgets commit nothing");
  const auto plan = layout.BackingPlan(1ULL << 30);
  Require(plan.size() == 1 && plan[0].block_bytes == 65536,
          "one page class serves rows and private state");
  Require(plan[0].blocks * plan[0].block_bytes + plan[0].metadata_bytes <=
              1ULL << 30,
          "backing stays inside the budget");
}
}  // namespace

int main() {
  try {
    TestInventory();
    TestPrivateBytes();
    TestPositions();
    TestRowsAndBacking();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
