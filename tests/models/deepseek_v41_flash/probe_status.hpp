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
  explicit ProbeStages(bool tokenization_selected = false)
      : tokenization_selected_(tokenization_selected) {}

  // A stage remains unknown until all its work is complete. A supplied
  // reason fails the stage. Stages with unmet dependencies remain blocked.
  // Return 1 for a failure or an unmet dependency; otherwise return 0.
  int Transition(Stage stage, std::uint64_t completed, std::uint64_t total,
                 const char* failure = nullptr) {
    if (failure)
      reason_ = failure;
    if (stage == Stage::kActive)
      stage = active_;
    switch (stage) {
      case Stage::kLoading:
        break;
      case Stage::kTokenization:
        if (Get(Stage::kLoading) != StageResult::kPass)
          return 1;
        break;
      case Stage::kPrefill:
        if (Get(Stage::kLoading) != StageResult::kPass ||
            (tokenization_selected_ &&
             Get(Stage::kTokenization) != StageResult::kPass))
          return 1;
        break;
      case Stage::kDecode:
        if (Get(Stage::kPrefill) != StageResult::kPass)
          return 1;
        break;
      default:
        return 1;
    }
    auto& result = results_[static_cast<std::size_t>(stage)];
    result = failure                            ? StageResult::kFail
             : total != 0 && completed == total ? StageResult::kPass
                                                : StageResult::kUnknown;
    active_ = result == StageResult::kUnknown ? stage : Stage::kActive;
    return failure ? 1 : 0;
  }

  [[nodiscard]] StageResult Get(Stage stage) const {
    const auto index = static_cast<std::size_t>(stage);
    return index < results_.size() ? results_[index] : StageResult::kUnknown;
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
  bool tokenization_selected_;
  std::array<StageResult, 4> results_{};
  Stage active_{Stage::kActive};
  const char* reason_{""};
};

}  // namespace gufo::tests::ds41
#endif
