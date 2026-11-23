#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::hy4_flash {

/// The Hy4-Flash pack tokenizer: the three `hy_v4` split stages, byte-level
/// BPE by merge rank, type-4 literals matched before BPE, and the pack aliases.
class Tokenizer {
public:
  static std::optional<Tokenizer> Load(const std::filesystem::path& directory,
                                       std::string* error);

  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  ~Tokenizer();

  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  [[nodiscard]] std::vector<int> Encode(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const int> ids) const;
  [[nodiscard]] int eos() const;

private:
  struct Impl;

  explicit Tokenizer(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::hy4_flash