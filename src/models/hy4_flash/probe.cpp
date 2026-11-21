#include "probe.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <gufo/ext/gguf_reader.hpp>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "pack.hpp"

namespace gufo::models::hy4_flash {
namespace {

constexpr std::string_view kArchitecture = "hy_v4_flash";
constexpr std::string_view kEngramTensor = "per_layer_token_embd.weight";
constexpr std::uint16_t kF8E4M3 = 43;
constexpr std::uint64_t kBlocks = 42;
constexpr std::uint64_t kExperts = 512;
constexpr std::uint64_t kDenseLead = 1;

bool Fail(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
  return false;
}

bool RequireTensorType(const core::GgufReader& reader, std::string_view name,
                       core::GgmlType type, std::string* error) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr) {
    return Fail(error, "missing tensor " + std::string(name));
  }
  if (tensor->type != type) {
    return Fail(error, std::string(name) + " is " +
                           std::string(core::ToString(tensor->type)) +
                           ", not " + std::string(core::ToString(type)));
  }
  return true;
}

bool ReadUintArray(const core::GgufReader& reader, std::string_view key,
                   std::vector<std::uint64_t>* values, std::string* error) {
  const auto* entry = reader.FindMetadata(key);
  if (entry == nullptr) {
    return Fail(error, "missing " + std::string(key));
  }
  if (const auto* unsigned_values =
          std::get_if<std::vector<std::uint64_t>>(&entry->value)) {
    *values = *unsigned_values;
    return true;
  }
  if (const auto* signed_values =
          std::get_if<std::vector<std::int64_t>>(&entry->value)) {
    values->clear();
    values->reserve(signed_values->size());
    for (const auto value : *signed_values) {
      if (value < 0) {
        return Fail(error, std::string(key) + " has a negative value");
      }
      values->push_back(static_cast<std::uint64_t>(value));
    }
    return true;
  }
  return Fail(error, std::string(key) + " is not an integer array");
}

bool CheckEngram(const core::GgufReader& reader, std::string* error) {
  const auto* tensor = reader.FindTensor(kEngramTensor);
  if (tensor == nullptr) {
    return Fail(error, "missing Engram tensor " + std::string(kEngramTensor));
  }
  if (tensor->sized || static_cast<std::uint16_t>(tensor->type) != kF8E4M3) {
    return Fail(error, "Engram tensor is not unsized type 43 (F8_E4M3)");
  }
  if (tensor->dimensions.size() != 2) {
    return Fail(error, "Engram tensor does not have two dimensions");
  }
  const auto width =
      reader.GetMetadataUint64("hy_v4_flash.embedding_length_per_layer_input");
  if (!width || *width == 0) {
    return Fail(error, "missing hy_v4_flash.embedding_length_per_layer_input");
  }
  if (tensor->dimensions[0] != *width) {
    return Fail(error, "Engram rows are not the per-layer embedding width");
  }
  const auto rows = tensor->dimensions[1];
  if (rows < 2) {
    return Fail(error, "Engram table has fewer than two rows");
  }
  const auto ngram =
      reader.GetMetadataUint64("hy_v4_flash.ple.ngram_size").value_or(0);
  if (ngram < 2) {
    return Fail(error, "Engram n-gram size is below two");
  }
  const auto heads =
      reader.GetMetadataUint64("hy_v4_flash.ple.heads_per_ngram").value_or(0);
  if (heads == 0) {
    return Fail(error, "Engram has no heads");
  }
  std::vector<std::uint64_t> layers;
  std::vector<std::uint64_t> multipliers;
  std::vector<std::uint64_t> offsets;
  std::vector<std::uint64_t> vocabularies;
  if (!ReadUintArray(reader, "hy_v4_flash.ple.layers", &layers, error) ||
      !ReadUintArray(reader, "hy_v4_flash.ple.layer_multipliers", &multipliers,
                     error) ||
      !ReadUintArray(reader, "hy_v4_flash.ple.head_offsets", &offsets, error) ||
      !ReadUintArray(reader, "hy_v4_flash.ple.head_vocab_sizes", &vocabularies,
                     error)) {
    return false;
  }
  const auto modules = layers.size();
  if (modules == 0) {
    return Fail(error, "Engram names no layers");
  }
  if (ngram - 1 > std::numeric_limits<std::uint64_t>::max() / heads) {
    return Fail(error, "Engram column count overflows");
  }
  const auto columns = (ngram - 1) * heads;
  if (modules > std::numeric_limits<std::uint64_t>::max() / columns ||
      modules > std::numeric_limits<std::uint64_t>::max() / ngram) {
    return Fail(error, "Engram head count overflows");
  }
  if (multipliers.size() != modules * ngram ||
      offsets.size() != modules * columns ||
      vocabularies.size() != modules * columns) {
    return Fail(error, "Engram array lengths do not match the module count");
  }
  for (std::size_t head = 0; head < offsets.size(); ++head) {
    const auto vocabulary = vocabularies[head];
    if (vocabulary < 2 || vocabulary > rows) {
      return Fail(error, "Engram head vocabulary is outside the table");
    }
    if (offsets[head] > rows - vocabulary) {
      return Fail(error, "Engram head rows are outside the table");
    }
  }
  if (rows > std::numeric_limits<std::uint64_t>::max() / *width) {
    return Fail(error, "Engram payload overflows");
  }
  if (*width * rows > tensor->size_bytes) {
    return Fail(error, "Engram tensor does not fit its reserved span");
  }
  return true;
}

bool CheckTrunk(const core::GgufReader& reader, std::string* error) {
  const auto block_count = reader.GetMetadataUint64("hy_v4_flash.block_count");
  if (block_count != kBlocks) {
    return Fail(error, "block count is not the selected model");
  }
  const auto experts = reader.GetMetadataUint64("hy_v4_flash.expert_count");
  if (experts != kExperts) {
    return Fail(error, "expert count is not the selected model");
  }
  const auto dense =
      reader.GetMetadataUint64("hy_v4_flash.leading_dense_block_count");
  if (dense != kDenseLead) {
    return Fail(error, "dense block count is not the selected model");
  }
  if (const auto mtp =
          reader.GetMetadataUint64("hy_v4_flash.nextn_predict_layers");
      mtp && *mtp != 0) {
    return Fail(error, "the set has an MTP block");
  }
  for (std::uint64_t layer = 0; layer < *block_count; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    if (!RequireTensorType(reader, prefix + "attn_q_a.weight",
                           core::GgmlType::kQ6_K, error) ||
        !RequireTensorType(reader, prefix + "attn_output.weight",
                           core::GgmlType::kQ6_K, error)) {
      return false;
    }
    if (layer < *dense) {
      for (const std::string_view weight :
           {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
        if (!RequireTensorType(reader, prefix + std::string(weight),
                               core::GgmlType::kQ6_K, error)) {
          return false;
        }
      }
      continue;
    }
    for (const std::string_view expert :
         {"ffn_gate_exps.weight", "ffn_up_exps.weight",
          "ffn_down_exps.weight"}) {
      if (!RequireTensorType(reader, prefix + std::string(expert),
                             core::GgmlType::kIQ2_XXS, error)) {
        return false;
      }
    }
    for (const std::string_view shared :
         {"ffn_gate_shexp.weight", "ffn_up_shexp.weight",
          "ffn_down_shexp.weight"}) {
      if (!RequireTensorType(reader, prefix + std::string(shared),
                             core::GgmlType::kQ8_0, error)) {
        return false;
      }
    }
  }
  return true;
}

bool CheckModel(const core::GgufReader& reader, std::string* error) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture != kArchitecture) {
    return Fail(error, "architecture is not " + std::string(kArchitecture));
  }
  if (const auto type = reader.GetMetadataString("general.type");
      type && !type->empty() && *type != "model") {
    return Fail(error, "file is a " + std::string(*type) + " sidecar");
  }
  if (const auto count = reader.GetMetadataUint32("split.count");
      !count || *count != 2) {
    return Fail(error, "the set is not the two-shard Maya-XS pair");
  }
  if (const auto number = reader.GetMetadataUint32("split.no");
      !number || *number != 0) {
    return Fail(error, "the first shard was not opened");
  }
  return CheckTrunk(reader, error) && CheckEngram(reader, error);
}

std::filesystem::path Canonical(const std::filesystem::path& path) {
  std::error_code error;
  const auto canonical = std::filesystem::weakly_canonical(path, error);
  return error ? path.lexically_normal() : canonical;
}

bool OpenModel(std::string_view model_root,
               std::unique_ptr<core::GgufReader>* reader, std::string* error) {
  const std::filesystem::path path(model_root);
  if (!std::filesystem::is_regular_file(path)) {
    return Fail(error, std::string(model_root) + ": not a file");
  }
  std::string open_error;
  *reader = core::GgufReader::OpenFile(path, &open_error,
                                       {.allow_unsized_tensors = true});
  if (!*reader) {
    return Fail(error, std::string(model_root) + ": " + open_error);
  }
  return true;
}

}  // namespace

bool ProbeHy4FlashGguf(const core::GgufReader& reader) {
  return CheckModel(reader, nullptr);
}

bool ProbeHy4Flash(std::string_view model_root) {
  try {
    std::unique_ptr<core::GgufReader> reader;
    std::string error;
    return OpenModel(model_root, &reader, &error) &&
           CheckModel(*reader, &error);
  } catch (const std::exception&) {
    return false;
  }
}

bool VerifyHy4FlashModel(std::string_view model_root, std::string* error) {
  try {
    std::unique_ptr<core::GgufReader> reader;
    if (!OpenModel(model_root, &reader, error) || !CheckModel(*reader, error)) {
      return false;
    }
    const auto path = std::filesystem::path(model_root);
    const auto regions = reader->GetMappedRegions();
    if (regions.empty() ||
        Canonical(regions.front().source_path) != Canonical(path)) {
      return Fail(error, "the first shard was not opened");
    }
    const auto pack = path.parent_path() / "maya-hy4";
    std::error_code exists_error;
    if (!std::filesystem::exists(pack, exists_error)) {
      return Fail(error, "missing pack " + pack.string());
    }
    return VerifyPack(pack, *reader, error);
  } catch (const std::exception& exception) {
    return Fail(error, exception.what());
  }
}

}  // namespace gufo::models::hy4_flash
