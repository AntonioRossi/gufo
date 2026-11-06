#pragma once

#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::deepseek_v41_flash {

/* Both variants use deepseek41. The expert count makes the probes disjoint:
 * the 384 probe rejects a Coder file, and the Coder probe rejects the 384
 * file and every unsupported count. */
[[nodiscard]] bool ProbeDeepSeekV384(std::string_view model_root);
[[nodiscard]] bool ProbeDeepSeekV384Gguf(const core::GgufReader& reader);
[[nodiscard]] bool ProbeDeepSeekCoder(std::string_view model_root);
[[nodiscard]] bool ProbeDeepSeekCoderGguf(const core::GgufReader& reader);

}  // namespace gufo::models::deepseek_v41_flash