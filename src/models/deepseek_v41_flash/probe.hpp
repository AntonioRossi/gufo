#pragma once

#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::deepseek_v41_flash {

[[nodiscard]] bool ProbeDeepSeek(std::string_view model_root);
[[nodiscard]] bool ProbeDeepSeekGguf(const core::GgufReader& reader);

}  // namespace gufo::models::deepseek_v41_flash
