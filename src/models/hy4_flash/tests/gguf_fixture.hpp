#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace hy4_test {

inline constexpr std::string_view kShard1Name =
    "Hy4-Fixture-00001-of-00002.gguf";
inline constexpr std::string_view kShard2Name =
    "Hy4-Fixture-00002-of-00002.gguf";

struct Fixture {
  std::string architecture{"hy_v4_flash"};
  std::string general_type;
  std::uint32_t split_count{2};
  std::uint32_t block_count{42};
  std::uint32_t expert_count{512};
  std::uint32_t dense_blocks{1};
  std::uint64_t width{96};
  std::uint32_t attention_type{14};
  std::uint32_t dense_ffn_type{14};
  std::uint32_t expert_type{16};
  std::uint32_t shared_type{8};
  std::uint32_t engram_type{43};
  std::uint64_t engram_width{0};
  std::uint64_t rows{8};
  std::int64_t module_layer{1};
  std::uint32_t ngram{2};
  std::uint32_t heads{2};
  std::vector<std::uint64_t> offsets{0, 4};
  std::vector<std::uint64_t> vocabularies{4, 4};
  std::vector<std::uint64_t> multipliers{1, 2};
  std::uint32_t nextn_predict_layers{0};
  bool include_engram{true};
  bool truncate_engram{false};
};

struct PackedTensor {
  std::string name;
  std::string type;
  std::uint32_t file{0};
  std::uint64_t offset{0};
  std::uint64_t bytes{0};
  std::vector<std::uint64_t> dimensions;
};

struct BuiltPair {
  std::vector<std::uint8_t> shard1;
  std::vector<std::uint8_t> shard2;
  std::vector<PackedTensor> tensors;
};

/// A two-shard set: the trunk in shard 1, the Engram table in shard 2.
BuiltPair BuildPair(const Fixture& fixture);

/// All tensors in one image with the split metadata, for metadata checks.
std::vector<std::uint8_t> BuildSingle(const Fixture& fixture);

bool WritePair(const std::filesystem::path& dir, const BuiltPair& pair,
               std::string* error);

/// Writes `hy4.txt` and `tensors.txt` from the same tensor list.
bool WritePack(const std::filesystem::path& dir, const BuiltPair& pair,
               std::string* error);

}  // namespace hy4_test
