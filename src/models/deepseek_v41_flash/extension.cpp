#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/deepseek_v41_flash/tool_calls.hpp"

namespace gufo::models::deepseek_v41_flash {

void RegisterToolDialect() {
  ext::RegisterToolDialect({
      .format = sampling::JsonConstraint::ToolFormat::kDeepSeek41,
      .markers = {DeepSeekToolSyntax::kCallsOpenWithSeparator,
                  DeepSeekToolSyntax::kCallsOpen},
      .reasoning_markers = {DeepSeekToolSyntax::kNativeCallOpenWithSeparator,
                            DeepSeekToolSyntax::kNativeCallOpen},
      .closers = {},
      .scan_tool_calls = &ScanDeepSeekToolCalls,
  });
}

}  // namespace gufo::models::deepseek_v41_flash

void deepseek_v41_flash_register(gufo::ext::Registry&) {
  gufo::models::deepseek_v41_flash::RegisterToolDialect();
}
