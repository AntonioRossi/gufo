#include "pack.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gufo::models::hy4_flash {
namespace {

constexpr std::string_view kArchitecture = "hy_v4_flash";
constexpr std::string_view kEngramTensor = "per_layer_token_embd.weight";
constexpr std::uint16_t kF8E4M3 = 43;

struct PackedTensor {
  std::string name;
  std::string type;
  std::uint64_t file{0};
  std::uint64_t offset{0};
  std::uint64_t bytes{0};
  std::vector<std::uint64_t> dimensions;
};

bool Fail(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
  return false;
}

bool ReadLines(const std::filesystem::path& path,
               std::vector<std::string>* lines, std::string* error) {
  std::ifstream input(path);
  if (!input) {
    return Fail(error, path.string() + ": cannot open");
  }
  for (std::string line; std::getline(input, line);) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    lines->push_back(std::move(line));
  }
  if (input.bad()) {
    return Fail(error, path.string() + ": read error");
  }
  return true;
}

std::vector<std::string> SplitFields(const std::string& line) {
  std::istringstream input(line);
  std::vector<std::string> fields;
  for (std::string field; input >> field;) {
    fields.push_back(std::move(field));
  }
  return fields;
}

bool ParseUint(std::string_view text, std::uint64_t* value) {
  if (text.empty()) {
    return false;
  }
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

std::filesystem::path Canonical(const std::filesystem::path& path) {
  std::error_code error;
  const auto canonical = std::filesystem::weakly_canonical(path, error);
  return error ? path.lexically_normal() : canonical;
}

std::filesystem::path Resolve(const std::string& model_dir,
                              const std::string& path) {
  const std::filesystem::path value(path);
  return value.is_absolute() ? value : std::filesystem::path(model_dir) / value;
}

std::string TypeName(const core::GgufTensorInfo& tensor) {
  if (static_cast<std::uint16_t>(tensor.type) == kF8E4M3) {
    return tensor.sized ? std::string() : std::string("F8_E4M3");
  }
  return std::string(core::ToString(tensor.type));
}

std::string PayloadProblem(const core::GgufTensorInfo& tensor,
                           const PackedTensor& packed) {
  if (tensor.sized) {
    if (packed.bytes != tensor.size_bytes) {
      return packed.name + ": payload size does not match";
    }
    return {};
  }
  if (packed.name != kEngramTensor || packed.type != "F8_E4M3" ||
      packed.dimensions.size() != 2) {
    return packed.name + ": unsupported unsized tensor";
  }
  if (packed.dimensions[0] == 0) {
    return packed.name + ": payload is empty";
  }
  if (packed.dimensions[1] >
      std::numeric_limits<std::uint64_t>::max() / packed.dimensions[0]) {
    return packed.name + ": payload overflows";
  }
  const auto payload = packed.dimensions[0] * packed.dimensions[1];
  if (packed.bytes != payload) {
    return packed.name + ": payload size does not match";
  }
  if (payload > tensor.size_bytes) {
    return packed.name + ": payload exceeds its reserved span";
  }
  return {};
}

enum class FileKind { kTrunk, kVision, kDrafter };

struct OpenedFile {
  const core::GgufReader* reader{nullptr};
  std::unique_ptr<core::GgufReader> owned;
  std::span<const core::GgufMappedRegion> regions;
  std::span<const core::GgufTensorInfo> tensors;
  FileKind kind{FileKind::kTrunk};
  std::vector<bool> seen;
  std::vector<bool> region_used;
};

struct RegionRef {
  std::size_t file{0};
  std::size_t region{0};
};

bool ClassifyCompanion(const core::GgufReader& reader, std::string_view path,
                       FileKind* kind, std::string* error) {
  const auto type = reader.GetMetadataString("general.type").value_or("");
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (type == "mtp") {
    return Fail(error, std::string(path) + ": MTP companion is not supported");
  }
  if (type == "mmproj" && architecture == "clip") {
    const auto projector =
        reader.GetMetadataString("clip.projector_type").value_or("");
    if (projector != "hy_v4_flash") {
      return Fail(error, std::string(path) +
                             ": clip.projector_type is not hy_v4_flash");
    }
    *kind = FileKind::kVision;
    return true;
  }
  if (type == "drafter" && architecture == "hy_v4_flash_dfly") {
    *kind = FileKind::kDrafter;
    return true;
  }
  return Fail(error, std::string(path) + ": unsupported companion (type " +
                         std::string(type) + ", architecture " +
                         std::string(architecture) + ")");
}

}  // namespace

bool VerifyPack(const std::filesystem::path& pack_dir,
                const core::GgufReader& reader, std::string* error) {
  std::vector<std::string> geometry;
  if (!ReadLines(pack_dir / "hy4.txt", &geometry, error)) {
    return false;
  }
  std::string model_dir;
  bool architecture = false;
  std::optional<std::uint64_t> vision_dim;
  std::optional<std::uint64_t> dfly_layers;
  std::size_t geometry_line = 0;
  const auto read_geometry = [&](std::string_view key,
                                 std::optional<std::uint64_t>* value,
                                 std::string_view text) {
    const auto prefix =
        pack_dir.string() + "/hy4.txt:" + std::to_string(geometry_line) + ": ";
    if (*value) {
      return Fail(error, prefix + "repeated " + std::string(key));
    }
    std::uint64_t parsed = 0;
    if (!ParseUint(text, &parsed)) {
      return Fail(error, prefix + "invalid " + std::string(key));
    }
    if (parsed == 0) {
      return Fail(error, prefix + std::string(key) + " must be positive");
    }
    *value = parsed;
    return true;
  };
  for (const auto& line : geometry) {
    ++geometry_line;
    const auto space = line.find(' ');
    const auto key = space == std::string::npos ? line : line.substr(0, space);
    const auto value =
        space == std::string::npos ? std::string() : line.substr(space + 1);
    if (key == "arch") {
      architecture = value == kArchitecture;
    } else if (key == "model_dir") {
      model_dir = value;
    } else if (key == "vision.dim") {
      if (!read_geometry("vision.dim", &vision_dim, value)) {
        return false;
      }
    } else if (key == "dfly_layers") {
      if (!read_geometry("dfly_layers", &dfly_layers, value)) {
        return false;
      }
    }
  }
  if (!architecture) {
    return Fail(error, pack_dir.string() + "/hy4.txt: not a hy_v4_flash pack");
  }
  if (model_dir.empty()) {
    return Fail(error, pack_dir.string() + "/hy4.txt: no model_dir");
  }

  std::vector<OpenedFile> files;
  {
    OpenedFile trunk;
    trunk.reader = &reader;
    trunk.regions = reader.GetMappedRegions();
    trunk.tensors = reader.GetTensors();
    trunk.seen.assign(trunk.tensors.size(), false);
    trunk.region_used.assign(trunk.regions.size(), false);
    files.push_back(std::move(trunk));
  }
  std::vector<RegionRef> region_of_file;
  std::unordered_set<std::string> tensor_names;

  const auto index_path = pack_dir / "tensors.txt";
  std::ifstream index_file(index_path);
  if (!index_file) {
    return Fail(error, index_path.string() + ": cannot open");
  }
  std::size_t line_number = 0;
  std::string line;
  while (std::getline(index_file, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const auto fields = SplitFields(line);
    if (fields.empty()) {
      continue;
    }
    const auto line_fail = [&](std::string reason) {
      return Fail(error, "tensors.txt:" + std::to_string(line_number) + ": " +
                             std::move(reason));
    };
    if (fields[0] == "file") {
      if (fields.size() != 3) {
        return line_fail("invalid field count");
      }
      std::uint64_t number = 0;
      if (!ParseUint(fields[1], &number)) {
        return line_fail("invalid number");
      }
      if (number != region_of_file.size()) {
        return line_fail("files out of order");
      }
      const auto canonical = Canonical(Resolve(model_dir, fields[2]));
      std::size_t file_index = files.size();
      std::size_t region_index = 0;
      bool found = false;
      for (std::size_t file = 0; file < files.size() && !found; ++file) {
        for (std::size_t region = 0; region < files[file].regions.size();
             ++region) {
          if (!files[file].region_used[region] &&
              Canonical(files[file].regions[region].source_path) == canonical) {
            file_index = file;
            region_index = region;
            found = true;
            break;
          }
        }
      }
      if (!found) {
        std::string open_error;
        auto companion = core::GgufReader::OpenFile(canonical, &open_error);
        if (!companion) {
          return line_fail("file not opened by Gufo: " + fields[2] + ": " +
                           open_error);
        }
        FileKind kind = FileKind::kTrunk;
        std::string classify_error;
        if (!ClassifyCompanion(*companion, fields[2], &kind, &classify_error)) {
          return line_fail(classify_error);
        }
        const auto companion_regions = companion->GetMappedRegions();
        if (companion_regions.size() != 1 ||
            Canonical(companion_regions[0].source_path) != canonical) {
          return line_fail("the companion must be one file: " + fields[2]);
        }
        OpenedFile opened;
        opened.reader = companion.get();
        opened.owned = std::move(companion);
        opened.regions = opened.reader->GetMappedRegions();
        opened.tensors = opened.reader->GetTensors();
        opened.kind = kind;
        opened.seen.assign(opened.tensors.size(), false);
        opened.region_used.assign(opened.regions.size(), false);
        files.push_back(std::move(opened));
        file_index = files.size() - 1;
        region_index = 0;
      }
      files[file_index].region_used[region_index] = true;
      region_of_file.push_back({file_index, region_index});
      continue;
    }
    if (fields[0] != "t") {
      return line_fail("unknown record");
    }
    if (fields.size() < 6) {
      return line_fail("missing field");
    }
    PackedTensor packed;
    packed.name = fields[1];
    packed.type = fields[2];
    if (!tensor_names.insert(packed.name).second) {
      return line_fail("the pack names a tensor twice: " + packed.name);
    }
    if (!ParseUint(fields[3], &packed.file) ||
        !ParseUint(fields[4], &packed.offset) ||
        !ParseUint(fields[5], &packed.bytes)) {
      return line_fail("invalid number");
    }
    if (packed.file >= region_of_file.size()) {
      return line_fail(packed.name + ": file index outside the pack");
    }
    const auto ref = region_of_file[packed.file];
    auto& file = files[ref.file];
    const auto* tensor = file.reader->FindTensor(packed.name);
    if (tensor == nullptr) {
      return line_fail("the pack names a tensor the GGUF lacks: " +
                       packed.name);
    }
    const auto rank = tensor->dimensions.size();
    if (fields.size() != 6 + rank) {
      return line_fail("invalid dimension");
    }
    for (std::size_t index = 0; index < rank; ++index) {
      std::uint64_t dimension = 0;
      if (!ParseUint(fields[6 + index], &dimension)) {
        return line_fail("invalid dimension");
      }
      packed.dimensions.push_back(dimension);
    }
    const auto index = static_cast<std::size_t>(tensor - file.tensors.data());
    file.seen[index] = true;
    const auto& region = file.regions[ref.region];
    if (packed.offset > region.size ||
        packed.bytes > region.size - packed.offset) {
      return line_fail(packed.name + ": outside its file");
    }
    const auto* address =
        static_cast<const std::uint8_t*>(region.data) + packed.offset;
    if (address != tensor->data) {
      return line_fail(packed.name + ": address does not match the GGUF");
    }
    if (packed.type != TypeName(*tensor)) {
      return line_fail(packed.name + ": type does not match the GGUF");
    }
    if (packed.dimensions != tensor->dimensions) {
      return line_fail(packed.name + ": dimensions do not match the GGUF");
    }
    const auto problem = PayloadProblem(*tensor, packed);
    if (!problem.empty()) {
      return line_fail(problem);
    }
  }
  if (index_file.bad()) {
    return Fail(error, "tensors.txt: read error");
  }

  for (const auto& file : files) {
    for (std::size_t region = 0; region < file.regions.size(); ++region) {
      if (!file.region_used[region]) {
        return Fail(error, "the pack omits file " +
                               file.regions[region].source_path.string());
      }
    }
    for (std::size_t index = 0; index < file.tensors.size(); ++index) {
      if (!file.seen[index]) {
        return Fail(error, "the pack omits tensor " +
                               std::string(file.tensors[index].name));
      }
    }
  }
  std::size_t visions = 0;
  std::size_t drafters = 0;
  const OpenedFile* vision = nullptr;
  const OpenedFile* drafter = nullptr;
  for (const auto& file : files) {
    if (file.kind == FileKind::kVision) {
      ++visions;
      vision = &file;
    } else if (file.kind == FileKind::kDrafter) {
      ++drafters;
      drafter = &file;
    }
  }
  if (visions == 0) {
    return Fail(error, "missing vision companion");
  }
  if (drafters == 0) {
    return Fail(error, "missing drafter companion");
  }
  if (visions != 1) {
    return Fail(error, "more than one vision companion");
  }
  if (drafters != 1) {
    return Fail(error, "more than one drafter companion");
  }
  if (region_of_file.size() != 4) {
    return Fail(error, "the pack must index four files");
  }
  if (!vision_dim) {
    return Fail(error, pack_dir.string() + "/hy4.txt: missing vision.dim");
  }
  const auto vision_width =
      vision->reader->GetMetadataUint64("clip.vision.embedding_length");
  if (!vision_width || *vision_width != *vision_dim) {
    return Fail(error, pack_dir.string() +
                           "/hy4.txt: vision.dim does not match the vision "
                           "encoder");
  }
  if (!dfly_layers) {
    return Fail(error, pack_dir.string() + "/hy4.txt: missing dfly_layers");
  }
  const auto drafter_layers =
      drafter->reader->GetMetadataUint64("hy_v4_flash_dfly.block_count");
  if (!drafter_layers || *drafter_layers != *dfly_layers) {
    return Fail(error, pack_dir.string() +
                           "/hy4.txt: dfly_layers does not match the drafter");
  }
  return true;
}

}  // namespace gufo::models::hy4_flash