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
inline constexpr std::string_view kVisionName = "Hy4-Fixture-mmproj.gguf";
inline constexpr std::string_view kVisionSplit1Name =
    "Hy4-Fixture-mmproj-00001-of-00002.gguf";
inline constexpr std::string_view kVisionSplit2Name =
    "Hy4-Fixture-mmproj-00002-of-00002.gguf";
inline constexpr std::string_view kDrafterName = "Hy4-Fixture-drafter.dfly2";
inline constexpr std::string_view kMtpName = "Hy4-Fixture-mtp.gguf";

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
  std::string vision_type{"mmproj"};
  std::string vision_architecture{"clip"};
  std::string vision_projector_type{"hy_v4_flash"};
  std::uint32_t vision_embedding_length{4};
  std::uint32_t vision_dim{4};
  std::uint32_t dfly_block_count{1};
  std::uint32_t dfly_layers{1};
  std::string companion_tensor_name{"vision.test.weight"};
  bool include_mtp{false};
  bool split_vision{false};
  std::string drafter_type{"drafter"};
  std::string drafter_architecture{"hy_v4_flash_dfly"};
};

struct PackOptions {
  bool include_companions{true};
  bool split_vision{false};
  bool include_mtp{false};
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
  std::vector<std::uint8_t> vision;
  std::vector<std::uint8_t> vision2;
  std::vector<std::uint8_t> drafter;
  std::vector<std::uint8_t> mtp;
  std::vector<PackedTensor> tensors;
  std::uint32_t vision_dim{4};
  std::uint32_t dfly_layers{1};
};

/// The two-shard trunk with the vision and drafter companions.
BuiltPair BuildPair(const Fixture& fixture);

/// All tensors in one image with the split metadata, for metadata checks.
std::vector<std::uint8_t> BuildSingle(const Fixture& fixture);

bool WritePair(const std::filesystem::path& dir, const BuiltPair& pair,
               std::string* error);

/// Writes `hy4.txt` and `tensors.txt` from the same tensor list. The options
/// select the companion files, the split vision set, and the MTP file.
bool WritePack(const std::filesystem::path& dir, const BuiltPair& pair,
               const PackOptions& options, std::string* error);

}  // namespace hy4_test
