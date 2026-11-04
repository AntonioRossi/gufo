#include "src/models/deepseek_v41_flash/tool_calls.hpp"

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "src/core/json.hpp"
#include "src/models/deepseek_v41_flash/chat_template.hpp"

namespace {

namespace ds41 = gufo::models::deepseek_v41_flash;
using gufo::ext::ChatRequest;
using gufo::json::parse;

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void CheckConversion() {
  std::ifstream file(GUFO_DS41_CHAT_GOLDENS);
  const auto fixture =
      parse(std::string(std::istreambuf_iterator<char>(file), {}));
  std::size_t checked = 0;
  for (const auto& golden : fixture.find("cases")->items()) {
    if (golden.find("tools")->empty() ||
        golden.member_str("name") == "image_tool_result")
      continue;
    ChatRequest request;
    request.reasoning.enabled = golden.find("thinking")->as_bool();
    const auto effort = golden.member_str("effort");
    request.reasoning.effort = effort == "low"   ? gufo::ReasoningEffort::kLow
                               : effort == "max" ? gufo::ReasoningEffort::kMax
                                                 : gufo::ReasoningEffort::kHigh;
    for (const auto& message : golden.find("messages")->items()) {
      auto& out = request.messages.emplace_back();
      const auto role = message.member_str("role");
      out.role = role == "system"      ? gufo::ext::ChatRole::kSystem
                 : role == "assistant" ? gufo::ext::ChatRole::kAssistant
                 : role == "tool"      ? gufo::ext::ChatRole::kTool
                 : role == "developer" ? gufo::ext::ChatRole::kDeveloper
                                       : gufo::ext::ChatRole::kUser;
      out.content = message.member_str("content");
      out.thought = message.member_str("reasoning_content");
      out.tool_call_id = message.member_str("tool_call_id");
      if (const auto* calls = message.find("tool_calls")) {
        for (const auto& call : calls->items()) {
          auto& converted = out.tool_calls.emplace_back();
          converted.id = call.member_str("id");
          converted.name = call.member_str("name");
          for (const auto& argument : call.find("arguments")->items())
            converted.arguments.push_back(
                {argument.member_str("name"), argument.member_str("value"),
                 argument.find("is_string")->as_bool()});
        }
      }
    }
    for (const auto& tool : golden.find("tools")->items()) {
      const auto definition = parse(tool.member_str("definition_json"));
      auto wrapper = gufo::json::Value::object();
      wrapper["type"] = "function";
      wrapper["function"] = definition;
      request.tools.push_back(
          {.name = tool.member_str("name"),
           .description = definition.member_str("description"),
           .parameters_json = definition.find("parameters")->dump(),
           .definition_json = wrapper.dump()});
    }
    request.constrained_tools = true;
    request.tool_choice = ChatRequest::ToolChoice::kRequired;
    request.forced_tool_name = request.tools.front().name;
    request.tool_namespaces[request.tools.front().name] = "functions";
    const auto converted = ds41::ConvertChat(request);
    Expect(ds41::RenderChat(converted.messages, converted.tools,
                            converted.options) == golden.member_str("expected"),
           "Converted tools must match the source fixture");
    for (std::size_t i = 0; i < request.messages.size(); ++i) {
      Expect(converted.messages[i].tool_call_id ==
                 request.messages[i].tool_call_id,
             "Keep the tool result ID");
      for (std::size_t j = 0; j < request.messages[i].tool_calls.size(); ++j)
        Expect(converted.messages[i].tool_calls[j].id ==
                   request.messages[i].tool_calls[j].id,
               "Keep the assistant call ID");
    }
    request.tool_choice = ChatRequest::ToolChoice::kNone;
    const auto disabled = ds41::ConvertChat(request);
    Expect(disabled.tools.empty() && !disabled.options.tools_present,
           "Disabled tools must not enter the template");
    request.messages.front().images.emplace_back();
    try {
      (void)ds41::ConvertChat(request);
      Expect(false, "Image input must fail");
    } catch (const std::invalid_argument&) {
    }
    ++checked;
  }
  Expect(checked == 4, "Check the four source tool fixtures");
}

bool Accepts(const gufo::sampling::JsonConstraint& grammar,
             std::string_view text) {
  auto state = grammar.Start();
  for (const unsigned char byte : text)
    state = grammar.Advance(state, byte);
  return grammar.Complete(state);
}

void CheckScannerAndGrammar() {
  const std::array markers{ds41::DeepSeekToolSyntax::kCallsOpen};
  const std::string call =
      "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n"
      "<｜DSML｜ parameter name=\"path\" string=\"true\">file.txt"
      "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
  std::vector<ds41::ScannedToolCall> calls;
  std::vector<std::pair<std::size_t, std::size_t>> spans;
  ds41::ScanDeepSeekToolCalls(call, markers, &calls, &spans);
  Expect(calls.size() == 1 && calls[0].name == "read" && calls[0].valid &&
             calls[0].complete,
         "Scan the completed source call");
  Expect(calls[0].arguments.size() == 1 &&
             calls[0].arguments[0].value == "file.txt" &&
             calls[0].arguments[0].is_string && spans.size() == 1 &&
             spans[0] == std::pair<std::size_t, std::size_t>{0, call.size()},
         "Keep the parameter value and block span");
  const auto close = call.find("</｜DSML｜ parameter>");
  for (std::size_t end = close;
       end < close + std::string("</｜DSML｜ parameter>").size(); ++end) {
    calls.clear();
    spans.clear();
    ds41::ScanDeepSeekToolCalls(std::string_view(call).substr(0, end), markers,
                                &calls, &spans, true);
    Expect(calls.size() == 1 && !calls[0].complete && calls[0].unfinished &&
               calls[0].unfinished->value == "file.txt" && spans.empty(),
           "Hold each partial parameter delimiter");
  }
  using Grammar = gufo::sampling::JsonConstraint;
  const auto schema = parse(
      R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false})");
  const auto arguments =
      Grammar::ToolParameters(schema, false, Grammar::ToolFormat::kDeepSeek41);
  const auto required =
      Grammar::WithTools(nullptr, {{"read", arguments}}, true, false,
                         Grammar::ToolFormat::kDeepSeek41);
  Expect(Accepts(*required, call) && !Accepts(*required, "DONE") &&
             !Accepts(*required, ""),
         "Required tool selection must use the V4.1 grammar");
}

}  // namespace

int main() {
  CheckConversion();
  CheckScannerAndGrammar();
}
