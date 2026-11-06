#include "src/models/deepseek_v41_flash/probe.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace gufo::models::deepseek_v41_flash {
namespace {

bool ProbeExpertCount(const core::GgufReader& reader, std::uint64_t expected) {
  if (reader.GetMetadataString("general.architecture").value_or("") !=
      "deepseek41")
    return false;
  return reader.GetMetadataUint64("deepseek41.expert_count").value_or(0) ==
         expected;
}

bool ProbeFile(std::string_view model_root, std::uint64_t expected) {
  try {
    const std::filesystem::path path(model_root);
    if (!std::filesystem::is_regular_file(path))
      return false;
    std::string error;
    auto reader = core::GgufReader::OpenFile(path.string(), &error);
    return reader && ProbeExpertCount(*reader, expected);
  } catch (...) {
    return false;
  }
}

}  // namespace

bool ProbeDeepSeekV384Gguf(const core::GgufReader& reader) {
  return ProbeExpertCount(reader, 384);
}

bool ProbeDeepSeekV384(std::string_view model_root) {
  return ProbeFile(model_root, 384);
}

bool ProbeDeepSeekCoderGguf(const core::GgufReader& reader) {
  return ProbeExpertCount(reader, 192);
}

bool ProbeDeepSeekCoder(std::string_view model_root) {
  return ProbeFile(model_root, 192);
}

}  // namespace gufo::models::deepseek_v41_flash