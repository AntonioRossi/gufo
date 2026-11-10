#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/deepseek_v41_flash/tool_calls.hpp"

namespace gufo::models::deepseek_v41_flash {

const sampling::JsonConstraint::ToolSyntax& DeepSeekToolGrammar() {
  static const sampling::JsonConstraint::ToolSyntax syntax{
      .open = std::string(DeepSeekToolSyntax::kCallsOpen),
      .name_prefix = "\n<｜DSML｜ invoke name=\"",
      .name_suffix = "\">\n",
      .close = "</｜DSML｜ invoke>",
      .block_close = "\n</｜DSML｜ calls>",
      .tagged =
          sampling::JsonConstraint::TaggedArguments{
              .parameter_open = "<｜DSML｜ parameter name=\"",
              .raw_value_open = "\" string=\"true\">",
              .typed_value_open = "\" string=\"false\">",
              .value_close = "</｜DSML｜ parameter>",
              .parameter_separator = "\n",
          },
      .reasoning_header = true,
  };
  return syntax;
}

void RegisterToolDialect() {
  ext::RegisterToolDialect({
      .markers = {DeepSeekToolSyntax::kCallsOpenWithSeparator,
                  DeepSeekToolSyntax::kCallsOpen},
      .reasoning_markers = {DeepSeekToolSyntax::kNativeCallOpenWithSeparator,
                            DeepSeekToolSyntax::kNativeCallOpen},
      .closers = {},
      .scan_tool_calls = &ScanDeepSeekToolCalls,
      .id = "deepseek41",
      .syntax = DeepSeekToolGrammar(),
  });
}

}  // namespace gufo::models::deepseek_v41_flash

void deepseek_v41_flash_register(gufo::ext::Registry&) {
  gufo::models::deepseek_v41_flash::RegisterToolDialect();
}
