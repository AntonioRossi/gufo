#include "pack.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
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

}  // namespace

bool VerifyPack(const std::filesystem::path& pack_dir,
                const core::GgufReader& reader, std::string* error) {
  std::vector<std::string> geometry;
  if (!ReadLines(pack_dir / "hy4.txt", &geometry, error)) {
    return false;
  }
  std::string model_dir;
  bool architecture = false;
  for (const auto& line : geometry) {
    const auto space = line.find(' ');
    if (space == std::string::npos) {
      continue;
    }
    const auto key = line.substr(0, space);
    const auto value = line.substr(space + 1);
    if (key == "arch") {
      architecture = value == kArchitecture;
    } else if (key == "model_dir") {
      model_dir = value;
    }
  }
  if (!architecture) {
    return Fail(error, pack_dir.string() + "/hy4.txt: not a hy_v4_flash pack");
  }
  if (model_dir.empty()) {
    return Fail(error, pack_dir.string() + "/hy4.txt: no model_dir");
  }

  const auto regions = reader.GetMappedRegions();
  const auto tensors = reader.GetTensors();
  std::vector<bool> seen(tensors.size(), false);
  std::vector<std::size_t> region_of_file;
  std::vector<bool> region_used(regions.size(), false);

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
      std::size_t region = regions.size();
      for (std::size_t index = 0; index < regions.size(); ++index) {
        if (!region_used[index] &&
            Canonical(regions[index].source_path) == canonical) {
          region = index;
          break;
        }
      }
      if (region == regions.size()) {
        return line_fail("file not opened by Gufo: " + fields[2]);
      }
      region_used[region] = true;
      region_of_file.push_back(region);
      continue;
    }
    if (fields[0] != "t") {
      return line_fail("unknown record");
    }
    if (fields.size() < 6) {
      return line_fail("missing field");
    }
    const auto* tensor = reader.FindTensor(fields[1]);
    if (tensor == nullptr) {
      return line_fail("the pack names a tensor the GGUF lacks: " + fields[1]);
    }
    const auto rank = tensor->dimensions.size();
    if (fields.size() != 6 + rank) {
      return line_fail("invalid dimension");
    }
    PackedTensor packed;
    packed.name = fields[1];
    packed.type = fields[2];
    if (!ParseUint(fields[3], &packed.file) ||
        !ParseUint(fields[4], &packed.offset) ||
        !ParseUint(fields[5], &packed.bytes)) {
      return line_fail("invalid number");
    }
    for (std::size_t index = 0; index < rank; ++index) {
      std::uint64_t dimension = 0;
      if (!ParseUint(fields[6 + index], &dimension)) {
        return line_fail("invalid dimension");
      }
      packed.dimensions.push_back(dimension);
    }
    if (packed.file >= region_of_file.size()) {
      return line_fail(packed.name + ": file index outside the pack");
    }
    const auto index = static_cast<std::size_t>(tensor - tensors.data());
    if (seen[index]) {
      return line_fail("the pack names a tensor twice: " + packed.name);
    }
    seen[index] = true;
    const auto& region = regions[region_of_file[packed.file]];
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

  if (region_of_file.size() != regions.size()) {
    return Fail(error,
                "the pack indexes " + std::to_string(region_of_file.size()) +
                    " files, Gufo opened " + std::to_string(regions.size()));
  }
  for (std::size_t index = 0; index < seen.size(); ++index) {
    if (!seen[index]) {
      return Fail(error,
                  "the pack omits tensor " + std::string(tensors[index].name));
    }
  }
  return true;
}

}  // namespace gufo::models::hy4_flash