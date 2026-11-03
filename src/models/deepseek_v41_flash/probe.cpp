#include "src/models/deepseek_v41_flash/probe.hpp"

#include <filesystem>
#include <string>

namespace gufo::models::deepseek_v41_flash {

bool ProbeDeepSeekGguf(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture").value_or("") ==
         "deepseek41";
}

bool ProbeDeepSeek(std::string_view model_root) {
  try {
    const std::filesystem::path path(model_root);
    if (!std::filesystem::is_regular_file(path))
      return false;
    std::string error;
    auto reader = core::GgufReader::OpenFile(path.string(), &error);
    return reader && ProbeDeepSeekGguf(*reader);
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::deepseek_v41_flash
