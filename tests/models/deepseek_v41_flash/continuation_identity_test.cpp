#include <iostream>
#include <stdexcept>
#include <string>

#include "tests/models/deepseek_v41_flash/identity_variants.hpp"

namespace {

using Bytes = std::vector<std::uint8_t>;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::string Text(const Bytes& bytes) {
  return {bytes.begin(), bytes.end()};
}

template<class Function>
void Reject(Function&& function, const std::string& message) {
  try {
    function();
  } catch (const std::invalid_argument& error) {
    Require(error.what() == message, "unexpected validation message");
    return;
  }
  throw std::runtime_error("invalid fingerprint accepted");
}

void CheckVariant(Bytes (*make)(std::string_view, std::string_view,
                                std::uint32_t, bool, std::uint32_t)) {
  const std::string artifact(64, 'a');
  const std::string draft(64, 'b');
  const auto base = make(artifact, draft, 524288, true, 7);
  const auto text = Text(base);
  Require(base == make(artifact, draft, 524288, true, 7),
          "identity is not stable");
  Require(
      text.find("schema=gufo-deepseek41-continuation-v1") != std::string::npos,
      "missing schema");
  Require(text.find("model_kind=deepseek41") != std::string::npos,
          "missing model kind");
  Require(text.find("variant=") != std::string::npos, "missing variant");
  Require(text.find("expert_count=") != std::string::npos,
          "missing expert count");
  Require(text.find("artifact_id=gguf-sha256-v1:") != std::string::npos,
          "missing artifact identity");
  Require(text.find("tokenizer=") != std::string::npos, "missing tokenizer");
  Require(text.find("chat_template=") != std::string::npos,
          "missing chat template");
  Require(text.find("payload_layout=") != std::string::npos, "missing layout");
  Require(text.find("state_abi=") != std::string::npos, "missing ABI");
  Require(text.find("context_tokens=524288") != std::string::npos,
          "missing context");
  Require(text.find("dspark=1") != std::string::npos, "missing DSpark mode");
  Require(text.find("max_draft_tokens=7") != std::string::npos,
          "missing draft limit");
  Require(text.find("support_id=gguf-sha256-v1:") != std::string::npos,
          "missing support identity");
  Require(make(std::string(64, 'c'), draft, 524288, true, 7) != base,
          "artifact change did not change identity");
  Require(make(artifact, std::string(64, 'd'), 524288, true, 7) != base,
          "drafter change did not change identity");
  Require(make(artifact, draft, 131072, true, 7) != base,
          "context change did not change identity");
  Require(make(artifact, draft, 524288, false, 7) != base,
          "DSpark change did not change identity");
  Require(make(artifact, draft, 524288, true, 5) != base,
          "draft-limit change did not change identity");
  for (const std::string& bad :
       {std::string{}, std::string(63, 'a'), std::string(64, 'A'),
        std::string(64, 'g'), std::string(65, 'a')}) {
    Reject([&] { make(bad, draft, 4096, true, 4); },
           "DeepSeek V4.1 disk cache requires an artifact fingerprint");
    Reject([&] { make(artifact, bad, 4096, true, 4); },
           "DeepSeek V4.1 DSpark disk cache requires a drafter fingerprint");
  }
}

}  // namespace

int main() {
  try {
    CheckVariant(identity_test::V384Identity);
    CheckVariant(identity_test::CoderIdentity);
    const auto v384 = identity_test::V384Identity(
        std::string(64, 'a'), std::string(64, 'b'), 524288, true, 7);
    const auto coder = identity_test::CoderIdentity(
        std::string(64, 'a'), std::string(64, 'b'), 524288, true, 7);
    Require(v384 != coder, "variant identities did not differ");
    std::cout << "continuation identity: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
