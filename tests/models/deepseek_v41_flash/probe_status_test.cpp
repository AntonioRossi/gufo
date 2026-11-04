#include "probe_status.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>

using gufo::tests::ds41::ProbeStages;
using gufo::tests::ds41::Stage;
using gufo::tests::ds41::StageResult;

void TestPromptCompletion(bool batch) {
  ProbeStages status;
  status.Transition(Stage::kLoading, 0, 1);
  status.Transition(Stage::kLoading, 1, 1);
  status.Transition(Stage::kPrefill, 0, 2);
  assert(status.Get(Stage::kPrefill) == StageResult::kUnknown);
  if (!batch) {
    status.Transition(Stage::kPrefill, 1, 2);
    assert(status.Get(Stage::kPrefill) == StageResult::kUnknown);
    assert(status.Get(Stage::kDecode) == StageResult::kBlocked);
  }
  status.Transition(Stage::kPrefill, 2, 2);
  assert(status.Get(Stage::kPrefill) == StageResult::kPass);
  status.Transition(Stage::kDecode, 0, 1);
  status.Transition(Stage::kDecode, 1, 1);
  assert(status.Get(Stage::kDecode) == StageResult::kPass);
  assert(std::strcmp(status.Reason(), "") == 0);
}

void TestSetupFailure() {
  ProbeStages status;
  status.Transition(Stage::kLoading, 0, 1);
  status.Transition(Stage::kLoading, 1, 1);
  status.Transition(Stage::kPrefill, 0, 2);
  status.Transition(Stage::kPrefill, 0, 2, "decoder_create_failed");
  assert(status.Get(Stage::kLoading) == StageResult::kPass);
  assert(status.Get(Stage::kPrefill) == StageResult::kFail);
  assert(status.Get(Stage::kDecode) == StageResult::kBlocked);
  assert(std::strcmp(status.Reason(), "decoder_create_failed") == 0);
}

void TestPartialFailure(bool exception) {
  ProbeStages status;
  status.Transition(Stage::kPrefill, 0, 2);
  status.Transition(Stage::kPrefill, 1, 2);
  if (exception) {
    try {
      throw std::runtime_error("test exception");
    } catch (const std::exception&) {
      status.Transition(Stage::kActive, 0, 1, "probe_exception");
    }
  } else {
    status.Transition(Stage::kPrefill, 1, 2, "prefill_failed");
  }
  assert(status.Get(Stage::kPrefill) == StageResult::kFail);
  assert(status.Get(Stage::kDecode) == StageResult::kBlocked);
  assert(std::strcmp(status.Reason(),
                     exception ? "probe_exception" : "prefill_failed") == 0);
}

int main() {
  TestPromptCompletion(false);
  TestPromptCompletion(true);
  TestSetupFailure();
  TestPartialFailure(false);
  TestPartialFailure(true);
  ProbeStages status;
  status.Transition(Stage::kLoading, 0, 1);
  status.Transition(Stage::kLoading, 0, 1, "engine_open_failed");
  assert(status.Get(Stage::kLoading) == StageResult::kFail);
  assert(status.Get(Stage::kPrefill) == StageResult::kBlocked);
  assert(std::strcmp(status.Reason(), "engine_open_failed") == 0);
}
