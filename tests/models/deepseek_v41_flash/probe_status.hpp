#ifndef GUFO_TESTS_DEEPSEEK_V41_FLASH_PROBE_STATUS_HPP_
#define GUFO_TESTS_DEEPSEEK_V41_FLASH_PROBE_STATUS_HPP_

#include <array>
#include <cstddef>
#include <cstdint>

namespace gufo::tests::ds41 {

enum class Stage : std::size_t {
  kLoading,
  kTokenization,
  kPrefill,
  kDecode,
  kActive
};
enum class StageResult { kBlocked, kUnknown, kPass, kFail };

class ProbeStages {
public:
  // A stage remains unknown until all its work is complete. A supplied
  // reason fails the stage. Stages with unmet dependencies remain blocked.
  void Transition(Stage stage, std::uint64_t completed, std::uint64_t total,
                  const char* failure = nullptr) {
    if (failure)
      reason_ = failure;
    if (stage == Stage::kActive)
      stage = active_;
    if (stage == Stage::kActive)
      return;
    auto& result = results_[static_cast<std::size_t>(stage)];
    result = failure                            ? StageResult::kFail
             : total != 0 && completed == total ? StageResult::kPass
                                                : StageResult::kUnknown;
    active_ = result == StageResult::kUnknown ? stage : Stage::kActive;
  }

  [[nodiscard]] StageResult Get(Stage stage) const {
    return results_[static_cast<std::size_t>(stage)];
  }
  [[nodiscard]] const char* Name(Stage stage) const {
    switch (Get(stage)) {
      case StageResult::kBlocked:
        return "blocked";
      case StageResult::kUnknown:
        return "unknown";
      case StageResult::kPass:
        return "pass";
      case StageResult::kFail:
        return "fail";
    }
    return "unknown";
  }
  [[nodiscard]] const char* Reason() const { return reason_; }

private:
  std::array<StageResult, 4> results_{};
  Stage active_{Stage::kActive};
  const char* reason_{""};
};

}  // namespace gufo::tests::ds41
#endif
