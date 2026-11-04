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
  assert(status.Get(Stage::kTokenization) == StageResult::kBlocked);
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
  status.Transition(Stage::kLoading, 0, 1);
  status.Transition(Stage::kLoading, 1, 1);
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

void TestInvalidAccess() {
  ProbeStages status;
  for (const auto stage : {Stage::kActive, static_cast<Stage>(99)}) {
    assert(status.Get(stage) == StageResult::kUnknown);
    assert(std::strcmp(status.Name(stage), "unknown") == 0);
  }
}

void TestExceptionWithoutActiveStage(bool after_pass) {
  ProbeStages status;
  if (after_pass) {
    status.Transition(Stage::kLoading, 1, 1);
    status.Transition(Stage::kPrefill, 2, 2);
    status.Transition(Stage::kDecode, 1, 1);
  }
  int result = 0;
  try {
    throw std::runtime_error("test exception");
  } catch (const std::exception&) {
    result = status.Transition(Stage::kActive, 0, 1, "probe_exception");
  }
  assert(result != 0);
  assert(std::strcmp(status.Reason(), "probe_exception") == 0);
  const auto expected = after_pass ? StageResult::kPass : StageResult::kBlocked;
  for (const auto stage : {Stage::kLoading, Stage::kPrefill, Stage::kDecode})
    assert(status.Get(stage) == expected);
  assert(status.Get(Stage::kTokenization) == StageResult::kBlocked);
}

void TestDependencies() {
  ProbeStages status(true);
  for (const auto stage :
       {Stage::kTokenization, Stage::kPrefill, Stage::kDecode}) {
    assert(status.Transition(stage, 1, 1) != 0);
    assert(status.Get(stage) == StageResult::kBlocked);
  }
  status.Transition(Stage::kLoading, 0, 1);
  assert(status.Transition(Stage::kTokenization, 1, 1) != 0);
  assert(status.Transition(Stage::kPrefill, 1, 1) != 0);
  status.Transition(Stage::kLoading, 1, 1);
  assert(status.Transition(Stage::kPrefill, 1, 1) != 0);
  status.Transition(Stage::kTokenization, 0, 1);
  assert(status.Transition(Stage::kPrefill, 1, 1) != 0);
  status.Transition(Stage::kTokenization, 1, 1);
  assert(status.Transition(Stage::kDecode, 1, 1) != 0);
  status.Transition(Stage::kPrefill, 0, 2);
  assert(status.Transition(Stage::kDecode, 1, 1) != 0);
  assert(status.Get(Stage::kPrefill) == StageResult::kUnknown);
  assert(status.Get(Stage::kDecode) == StageResult::kBlocked);
  assert(status.Transition(Stage::kPrefill, 2, 2) == 0);
  assert(status.Transition(Stage::kDecode, 1, 1) == 0);
  assert(status.Get(Stage::kDecode) == StageResult::kPass);
}

void TestTokenizationFailure() {
  ProbeStages status(true);
  status.Transition(Stage::kLoading, 1, 1);
  status.Transition(Stage::kTokenization, 0, 1);
  assert(status.Transition(Stage::kTokenization, 0, 1, "tokenization_failed") !=
         0);
  assert(status.Get(Stage::kTokenization) == StageResult::kFail);
  assert(status.Transition(Stage::kPrefill, 2, 2) != 0);
  assert(status.Transition(Stage::kDecode, 1, 1) != 0);
  assert(status.Get(Stage::kPrefill) == StageResult::kBlocked);
  assert(status.Get(Stage::kDecode) == StageResult::kBlocked);
  assert(std::strcmp(status.Reason(), "tokenization_failed") == 0);
}

void TestDecodeFailure() {
  ProbeStages status;
  status.Transition(Stage::kLoading, 1, 1);
  status.Transition(Stage::kPrefill, 2, 2);
  status.Transition(Stage::kDecode, 0, 1);
  assert(status.Transition(Stage::kDecode, 0, 1, "decode_failed") != 0);
  assert(status.Get(Stage::kPrefill) == StageResult::kPass);
  assert(status.Get(Stage::kDecode) == StageResult::kFail);
  assert(std::strcmp(status.Reason(), "decode_failed") == 0);
}

void TestInvalidTransition() {
  ProbeStages status;
  assert(status.Transition(static_cast<Stage>(99), 1, 1) != 0);
  for (const auto stage :
       {Stage::kLoading, Stage::kTokenization, Stage::kPrefill, Stage::kDecode})
    assert(status.Get(stage) == StageResult::kBlocked);
  assert(std::strcmp(status.Reason(), "") == 0);
}

void TestNoActiveTransition(bool after_pass) {
  ProbeStages status;
  if (after_pass)
    status.Transition(Stage::kLoading, 1, 1);
  assert(status.Transition(Stage::kActive, 1, 1) != 0);
  assert(status.Get(Stage::kLoading) ==
         (after_pass ? StageResult::kPass : StageResult::kBlocked));
  for (const auto stage :
       {Stage::kTokenization, Stage::kPrefill, Stage::kDecode})
    assert(status.Get(stage) == StageResult::kBlocked);
  assert(std::strcmp(status.Reason(), "") == 0);
}

void TestFailureWithUnmetDependency() {
  ProbeStages status;
  status.Transition(Stage::kLoading, 1, 1);
  status.Transition(Stage::kPrefill, 0, 2);
  assert(status.Transition(Stage::kDecode, 0, 1, "decode_failed") != 0);
  assert(status.Get(Stage::kPrefill) == StageResult::kUnknown);
  assert(status.Get(Stage::kDecode) == StageResult::kBlocked);
  assert(std::strcmp(status.Reason(), "decode_failed") == 0);
}

void TestCompletedResults() {
  const Stage stages[] = {Stage::kLoading, Stage::kTokenization,
                          Stage::kPrefill, Stage::kDecode};
  for (const auto stage : stages) {
    for (const bool failed : {false, true}) {
      ProbeStages status(true);
      for (const auto prerequisite : stages) {
        if (prerequisite == stage)
          break;
        assert(status.Transition(prerequisite, 1, 1) == 0);
      }
      const int expected = failed ? 1 : 0;
      const auto result = failed ? StageResult::kFail : StageResult::kPass;
      assert(status.Transition(stage, 1, 1,
                               failed ? "first_failure" : nullptr) == expected);
      assert(status.Transition(stage, 0, 2) == expected);
      assert(status.Get(stage) == result);
      assert(status.Transition(stage, 1, 1) == expected);
      assert(status.Get(stage) == result);
      assert(status.Transition(stage, 0, 1, "later_failure") != 0);
      assert(status.Get(stage) == result);
      assert(std::strcmp(status.Reason(), "later_failure") == 0);
    }
  }
}

int main() {
  TestPromptCompletion(false);
  TestPromptCompletion(true);
  TestSetupFailure();
  TestPartialFailure(false);
  TestPartialFailure(true);
  TestInvalidAccess();
  TestExceptionWithoutActiveStage(false);
  TestExceptionWithoutActiveStage(true);
  TestDependencies();
  TestTokenizationFailure();
  TestDecodeFailure();
  TestInvalidTransition();
  TestNoActiveTransition(false);
  TestNoActiveTransition(true);
  TestFailureWithUnmetDependency();
  TestCompletedResults();
  ProbeStages status;
  status.Transition(Stage::kLoading, 0, 1);
  status.Transition(Stage::kLoading, 0, 1, "engine_open_failed");
  assert(status.Get(Stage::kLoading) == StageResult::kFail);
  assert(status.Get(Stage::kPrefill) == StageResult::kBlocked);
  assert(std::strcmp(status.Reason(), "engine_open_failed") == 0);
}
