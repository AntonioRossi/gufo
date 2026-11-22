#include "gguf_fixture.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hy4_test {
namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747;
constexpr std::uint64_t kAlignment = 32;
constexpr std::uint32_t kTypeF32 = 0;
constexpr std::uint32_t kTypeF16 = 1;
constexpr std::uint32_t kTypeQ8_0 = 8;
constexpr std::uint32_t kTypeQ4_K = 12;
constexpr std::uint32_t kTypeQ6_K = 14;
constexpr std::uint32_t kTypeIQ2_XXS = 16;
constexpr std::uint32_t kTypeF8E4M3 = 43;
constexpr std::string_view kEngramTensor = "per_layer_token_embd.weight";

void PutU16(std::vector<std::uint8_t>& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}

void PutI32(std::vector<std::uint8_t>& out, std::int32_t value) {
  const auto bits = static_cast<std::uint32_t>(value);
  for (int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<std::uint8_t>(bits >> shift));
  }
}

void PutU32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void PutU64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void PutString(std::vector<std::uint8_t>& out, std::string_view value) {
  PutU64(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}

struct Meta {
  std::string key;
  std::vector<std::uint8_t> encoded;
};

void AddString(std::vector<Meta>& meta, std::string key,
               std::string_view value) {
  std::vector<std::uint8_t> encoded;
  PutU32(encoded, 8);
  PutString(encoded, value);
  meta.push_back({std::move(key), std::move(encoded)});
}

void AddU16(std::vector<Meta>& meta, std::string key, std::uint16_t value) {
  std::vector<std::uint8_t> encoded;
  PutU32(encoded, 2);
  PutU16(encoded, value);
  meta.push_back({std::move(key), std::move(encoded)});
}

void AddU32(std::vector<Meta>& meta, std::string key, std::uint32_t value) {
  std::vector<std::uint8_t> encoded;
  PutU32(encoded, 4);
  PutU32(encoded, value);
  meta.push_back({std::move(key), std::move(encoded)});
}

void AddI32(std::vector<Meta>& meta, std::string key, std::int32_t value) {
  std::vector<std::uint8_t> encoded;
  PutU32(encoded, 5);
  PutI32(encoded, value);
  meta.push_back({std::move(key), std::move(encoded)});
}

void AddI32Array(std::vector<Meta>& meta, std::string key,
                 const std::vector<std::int32_t>& values) {
  std::vector<std::uint8_t> encoded;
  PutU32(encoded, 9);
  PutU32(encoded, 5);
  PutU64(encoded, values.size());
  for (const auto value : values) {
    PutI32(encoded, value);
  }
  meta.push_back({std::move(key), std::move(encoded)});
}

void AddU64Array(std::vector<Meta>& meta, std::string key,
                 const std::vector<std::uint64_t>& values) {
  std::vector<std::uint8_t> encoded;
  PutU32(encoded, 9);
  PutU32(encoded, 10);
  PutU64(encoded, values.size());
  for (const auto value : values) {
    PutU64(encoded, value);
  }
  meta.push_back({std::move(key), std::move(encoded)});
}

struct TensorSpec {
  std::string name;
  std::uint32_t type;
  std::vector<std::uint64_t> dimensions;
  std::uint32_t file;
};

std::uint64_t Elements(const std::vector<std::uint64_t>& dimensions) {
  std::uint64_t count = 1;
  for (const auto dimension : dimensions) {
    count *= dimension;
  }
  return count;
}

std::uint64_t TensorBytes(std::uint32_t type,
                          const std::vector<std::uint64_t>& dimensions) {
  const auto elements = Elements(dimensions);
  switch (type) {
    case kTypeF32:
      return elements * 4;
    case kTypeF16:
      return elements * 2;
    case kTypeQ8_0:
      return elements / 32 * 34;
    case kTypeQ4_K:
      return elements / 256 * 144;
    case kTypeQ6_K:
      return elements / 256 * 210;
    case kTypeIQ2_XXS:
      return elements / 256 * 66;
    case kTypeF8E4M3:
      return elements;
    default:
      return 0;
  }
}

std::string TypeName(std::uint32_t type) {
  switch (type) {
    case kTypeF32:
      return "F32";
    case kTypeF16:
      return "F16";
    case kTypeQ8_0:
      return "Q8_0";
    case kTypeQ4_K:
      return "Q4_K";
    case kTypeQ6_K:
      return "Q6_K";
    case kTypeIQ2_XXS:
      return "IQ2_XXS";
    case kTypeF8E4M3:
      return "F8_E4M3";
    default:
      return {};
  }
}

void AddTensor(std::vector<TensorSpec>* tensors, std::string name,
               std::uint32_t type, std::vector<std::uint64_t> dimensions,
               std::uint32_t file) {
  tensors->push_back({std::move(name), type, std::move(dimensions), file});
}

std::vector<TensorSpec> MakeTensors(const Fixture& fixture,
                                    bool include_companions) {
  std::vector<TensorSpec> tensors;
  for (std::uint32_t layer = 0; layer < fixture.block_count; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    AddTensor(&tensors, prefix + "attn_q_a.weight", fixture.attention_type,
              {256, 1}, 0);
    AddTensor(&tensors, prefix + "attn_output.weight", fixture.attention_type,
              {256, 1}, 0);
    if (layer < fixture.dense_blocks) {
      for (const std::string_view weight :
           {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
        AddTensor(&tensors, prefix + std::string(weight),
                  fixture.dense_ffn_type, {256, 1}, 0);
      }
      continue;
    }
    for (const std::string_view expert :
         {"ffn_gate_exps.weight", "ffn_up_exps.weight",
          "ffn_down_exps.weight"}) {
      AddTensor(&tensors, prefix + std::string(expert), fixture.expert_type,
                {256, 1}, 0);
    }
    for (const std::string_view shared :
         {"ffn_gate_shexp.weight", "ffn_up_shexp.weight",
          "ffn_down_shexp.weight"}) {
      AddTensor(&tensors, prefix + std::string(shared), fixture.shared_type,
                {256, 1}, 0);
    }
  }
  AddTensor(&tensors, "output.weight", kTypeF32, {4}, 0);
  if (fixture.include_engram) {
    const auto width =
        fixture.engram_width != 0 ? fixture.engram_width : fixture.width;
    AddTensor(&tensors, std::string(kEngramTensor), fixture.engram_type,
              {width, fixture.rows}, 1);
  }
  if (include_companions) {
    if (fixture.split_vision) {
      AddTensor(&tensors, "vision.0.weight", kTypeF32, {4}, 2);
      AddTensor(&tensors, "vision.1.weight", kTypeF32, {4}, 3);
    } else {
      AddTensor(&tensors, fixture.companion_tensor_name, kTypeF32, {4}, 2);
    }
    AddTensor(&tensors, "drafter.test.weight", kTypeF32, {4},
              fixture.split_vision ? 4 : 3);
    if (fixture.include_mtp) {
      AddTensor(&tensors, "mtp.test.weight", kTypeF32, {4},
                fixture.split_vision ? 5 : 4);
    }
  }
  return tensors;
}

std::uint64_t Align(std::uint64_t value) {
  return (value + kAlignment - 1) / kAlignment * kAlignment;
}

bool FailWrite(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
  return false;
}

bool WriteFile(const std::filesystem::path& path,
               const std::vector<std::uint8_t>& bytes, std::string* error) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    return FailWrite(error, path.string() + ": cannot open");
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output) {
    return FailWrite(error, path.string() + ": write failed");
  }
  return true;
}

std::vector<std::uint8_t> Header(const std::vector<Meta>& meta,
                                 const std::vector<TensorSpec>& tensors,
                                 const std::vector<std::uint64_t>& offsets) {
  std::vector<std::uint8_t> out;
  PutU32(out, kGgufMagic);
  PutU32(out, 3);
  PutU64(out, tensors.size());
  PutU64(out, meta.size());
  for (const auto& entry : meta) {
    PutString(out, entry.key);
    out.insert(out.end(), entry.encoded.begin(), entry.encoded.end());
  }
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    PutString(out, tensors[index].name);
    PutU32(out, static_cast<std::uint32_t>(tensors[index].dimensions.size()));
    for (const auto dimension : tensors[index].dimensions) {
      PutU64(out, dimension);
    }
    PutU32(out, tensors[index].type);
    PutU64(out, offsets[index]);
  }
  return out;
}

std::vector<std::uint8_t> BuildContainer(
    std::vector<Meta> meta, const std::vector<TensorSpec>& tensors,
    bool truncate,
    std::vector<std::pair<std::string, std::uint64_t>>* absolute) {
  auto sorted = tensors;
  std::sort(sorted.begin(), sorted.end(),
            [](const TensorSpec& left, const TensorSpec& right) {
              return left.name < right.name;
            });
  std::sort(meta.begin(), meta.end(), [](const Meta& left, const Meta& right) {
    return left.key < right.key;
  });

  std::vector<std::uint64_t> offsets(sorted.size(), 0);
  const auto provisional = Header(meta, sorted, offsets);
  const auto data_start = Align(provisional.size());

  std::vector<std::uint64_t> sizes(sorted.size(), 0);
  std::uint64_t offset = 0;
  for (std::size_t index = 0; index < sorted.size(); ++index) {
    sizes[index] = TensorBytes(sorted[index].type, sorted[index].dimensions);
    offsets[index] = offset;
    offset = Align(offset + sizes[index]);
  }
  std::uint64_t data_bytes = offset;
  if (truncate) {
    for (std::size_t index = 0; index < sorted.size(); ++index) {
      if (sorted[index].name == kEngramTensor && sizes[index] != 0) {
        data_bytes = offsets[index] + sizes[index] / 2;
        break;
      }
    }
  }

  auto bytes = Header(meta, sorted, offsets);
  bytes.resize(data_start + data_bytes, 0);
  if (absolute != nullptr) {
    absolute->clear();
    for (std::size_t index = 0; index < sorted.size(); ++index) {
      absolute->push_back({sorted[index].name, data_start + offsets[index]});
    }
  }
  return bytes;
}

std::vector<std::uint8_t> BuildShard(
    const std::vector<TensorSpec>& tensors, const Fixture& fixture,
    std::uint32_t split_no, std::uint64_t total_tensors, bool truncate,
    std::vector<std::pair<std::string, std::uint64_t>>* absolute) {
  std::vector<Meta> meta;
  AddString(meta, "general.architecture", fixture.architecture);
  if (!fixture.general_type.empty()) {
    AddString(meta, "general.type", fixture.general_type);
  }
  AddU32(meta, "hy_v4_flash.block_count", fixture.block_count);
  AddU32(meta, "hy_v4_flash.embedding_length_per_layer_input",
         static_cast<std::uint32_t>(fixture.width));
  AddU32(meta, "hy_v4_flash.expert_count", fixture.expert_count);
  AddU32(meta, "hy_v4_flash.leading_dense_block_count", fixture.dense_blocks);
  if (fixture.nextn_predict_layers != 0) {
    AddU32(meta, "hy_v4_flash.nextn_predict_layers",
           fixture.nextn_predict_layers);
  }
  AddU64Array(meta, "hy_v4_flash.ple.head_offsets", fixture.offsets);
  AddU64Array(meta, "hy_v4_flash.ple.head_vocab_sizes", fixture.vocabularies);
  AddU32(meta, "hy_v4_flash.ple.heads_per_ngram", fixture.heads);
  AddI32Array(meta, "hy_v4_flash.ple.layers",
              {static_cast<std::int32_t>(fixture.module_layer)});
  AddU64Array(meta, "hy_v4_flash.ple.layer_multipliers", fixture.multipliers);
  AddU32(meta, "hy_v4_flash.ple.ngram_size", fixture.ngram);
  AddU16(meta, "split.count", static_cast<std::uint16_t>(fixture.split_count));
  AddU16(meta, "split.no", static_cast<std::uint16_t>(split_no));
  AddI32(meta, "split.tensors.count", static_cast<std::int32_t>(total_tensors));
  return BuildContainer(std::move(meta), tensors, truncate, absolute);
}

std::vector<std::uint8_t> BuildCompanion(
    const std::vector<TensorSpec>& tensors, std::vector<Meta> meta,
    std::vector<std::pair<std::string, std::uint64_t>>* absolute) {
  return BuildContainer(std::move(meta), tensors, false, absolute);
}

}  // namespace

BuiltPair BuildPair(const Fixture& fixture) {
  auto tensors = MakeTensors(fixture, true);
  std::vector<TensorSpec> trunk;
  std::vector<TensorSpec> engram;
  std::vector<TensorSpec> vision;
  std::vector<TensorSpec> vision2;
  std::vector<TensorSpec> drafter;
  std::vector<TensorSpec> mtp;
  for (const auto& tensor : tensors) {
    if (tensor.file == 0) {
      trunk.push_back(tensor);
    } else if (tensor.file == 1) {
      engram.push_back(tensor);
    } else if (tensor.file == 2) {
      vision.push_back(tensor);
    } else if (tensor.file == 3) {
      (fixture.split_vision ? vision2 : drafter).push_back(tensor);
    } else if (tensor.file == 4) {
      (fixture.split_vision ? drafter : mtp).push_back(tensor);
    } else {
      mtp.push_back(tensor);
    }
  }
  const auto split_tensors = trunk.size() + engram.size();
  BuiltPair pair;
  pair.vision_dim = fixture.vision_dim;
  pair.dfly_layers = fixture.dfly_layers;
  std::vector<std::pair<std::string, std::uint64_t>> trunk_offsets;
  std::vector<std::pair<std::string, std::uint64_t>> engram_offsets;
  std::vector<std::pair<std::string, std::uint64_t>> vision_offsets;
  std::vector<std::pair<std::string, std::uint64_t>> vision2_offsets;
  std::vector<std::pair<std::string, std::uint64_t>> drafter_offsets;
  std::vector<std::pair<std::string, std::uint64_t>> mtp_offsets;
  pair.shard1 =
      BuildShard(trunk, fixture, 0, split_tensors, false, &trunk_offsets);
  pair.shard2 = BuildShard(engram, fixture, 1, split_tensors,
                           fixture.truncate_engram, &engram_offsets);

  std::vector<Meta> vision_meta;
  AddString(vision_meta, "general.architecture", fixture.vision_architecture);
  AddString(vision_meta, "general.type", fixture.vision_type);
  AddString(vision_meta, "clip.projector_type", fixture.vision_projector_type);
  AddU32(vision_meta, "clip.vision.embedding_length",
         fixture.vision_embedding_length);
  if (fixture.split_vision) {
    AddU16(vision_meta, "split.count", 2);
    AddU16(vision_meta, "split.no", 0);
    AddI32(vision_meta, "split.tensors.count",
           static_cast<std::int32_t>(vision.size() + vision2.size()));
  }
  pair.vision = BuildCompanion(vision, vision_meta, &vision_offsets);
  if (fixture.split_vision) {
    std::vector<Meta> vision2_meta;
    AddString(vision2_meta, "general.architecture",
              fixture.vision_architecture);
    AddString(vision2_meta, "general.type", fixture.vision_type);
    AddString(vision2_meta, "clip.projector_type",
              fixture.vision_projector_type);
    AddU32(vision2_meta, "clip.vision.embedding_length",
           fixture.vision_embedding_length);
    AddU16(vision2_meta, "split.count", 2);
    AddU16(vision2_meta, "split.no", 1);
    AddI32(vision2_meta, "split.tensors.count",
           static_cast<std::int32_t>(vision.size() + vision2.size()));
    pair.vision2 = BuildCompanion(vision2, vision2_meta, &vision2_offsets);
  }

  std::vector<Meta> drafter_meta;
  AddString(drafter_meta, "general.architecture", fixture.drafter_architecture);
  AddString(drafter_meta, "general.type", fixture.drafter_type);
  AddU32(drafter_meta, "hy_v4_flash_dfly.block_count",
         fixture.dfly_block_count);
  pair.drafter = BuildCompanion(drafter, drafter_meta, &drafter_offsets);

  if (fixture.include_mtp) {
    std::vector<Meta> mtp_meta;
    AddString(mtp_meta, "general.architecture", "hy_v4_flash");
    AddString(mtp_meta, "general.type", "mtp");
    pair.mtp = BuildCompanion(mtp, mtp_meta, &mtp_offsets);
  }

  for (const auto& tensor : tensors) {
    const auto& offsets =
        tensor.file == 0   ? trunk_offsets
        : tensor.file == 1 ? engram_offsets
        : tensor.file == 2 ? vision_offsets
        : tensor.file == 3
            ? (fixture.split_vision ? vision2_offsets : drafter_offsets)
        : tensor.file == 4
            ? (fixture.split_vision ? drafter_offsets : mtp_offsets)
            : mtp_offsets;
    const auto found = std::find_if(
        offsets.begin(), offsets.end(),
        [&](const auto& entry) { return entry.first == tensor.name; });
    assert(found != offsets.end());
    pair.tensors.push_back(
        {tensor.name, TypeName(tensor.type), tensor.file, found->second,
         TensorBytes(tensor.type, tensor.dimensions), tensor.dimensions});
  }
  return pair;
}

std::vector<std::uint8_t> BuildSingle(const Fixture& fixture) {
  auto tensors = MakeTensors(fixture, false);
  std::vector<std::pair<std::string, std::uint64_t>> offsets;
  return BuildShard(tensors, fixture, 0, tensors.size(),
                    fixture.truncate_engram, &offsets);
}

bool WritePair(const std::filesystem::path& dir, const BuiltPair& pair,
               std::string* error) {
  const auto vision_name =
      pair.vision2.empty() ? kVisionName : kVisionSplit1Name;
  if (!WriteFile(dir / kShard1Name, pair.shard1, error) ||
      !WriteFile(dir / kShard2Name, pair.shard2, error) ||
      !WriteFile(dir / vision_name, pair.vision, error) ||
      !WriteFile(dir / kDrafterName, pair.drafter, error)) {
    return false;
  }
  if (!pair.vision2.empty() &&
      !WriteFile(dir / kVisionSplit2Name, pair.vision2, error)) {
    return false;
  }
  if (!pair.mtp.empty() && !WriteFile(dir / kMtpName, pair.mtp, error)) {
    return false;
  }
  return true;
}

bool WritePack(const std::filesystem::path& dir, const BuiltPair& pair,
               const PackOptions& options, std::string* error) {
  const auto pack = dir / "maya-hy4";
  std::error_code create_error;
  std::filesystem::create_directories(pack, create_error);
  if (create_error) {
    return FailWrite(error, pack.string() + ": " + create_error.message());
  }
  {
    std::ofstream geometry(pack / "hy4.txt");
    if (!geometry) {
      return FailWrite(error, (pack / "hy4.txt").string() + ": cannot open");
    }
    geometry << "arch hy_v4_flash\nmodel_dir "
             << std::filesystem::absolute(dir).string() << "\n"
             << "vision.dim " << pair.vision_dim << "\n"
             << "dfly_layers " << pair.dfly_layers << "\n";
    geometry.close();
    if (!geometry) {
      return FailWrite(error, (pack / "hy4.txt").string() + ": write failed");
    }
  }
  std::ofstream index(pack / "tensors.txt");
  if (!index) {
    return FailWrite(error, (pack / "tensors.txt").string() + ": cannot open");
  }
  index << "file 0 " << kShard1Name << "\n";
  index << "file 1 " << kShard2Name << "\n";
  if (options.include_companions) {
    index << "file 2 "
          << (options.split_vision ? kVisionSplit1Name : kVisionName) << "\n";
    if (options.split_vision) {
      index << "file 3 " << kVisionSplit2Name << "\n";
      index << "file 4 " << kDrafterName << "\n";
      if (options.include_mtp) {
        index << "file 5 " << kMtpName << "\n";
      }
    } else {
      index << "file 3 " << kDrafterName << "\n";
      if (options.include_mtp) {
        index << "file 4 " << kMtpName << "\n";
      }
    }
  }
  for (const auto& tensor : pair.tensors) {
    if (!options.include_companions && tensor.file >= 2) {
      continue;
    }
    index << "t " << tensor.name << ' ' << tensor.type << ' ' << tensor.file
          << ' ' << tensor.offset << ' ' << tensor.bytes;
    for (const auto dimension : tensor.dimensions) {
      index << ' ' << dimension;
    }
    index << '\n';
  }
  index.close();
  if (!index) {
    return FailWrite(error, (pack / "tensors.txt").string() + ": write failed");
  }
  return true;
}

}  // namespace hy4_test
