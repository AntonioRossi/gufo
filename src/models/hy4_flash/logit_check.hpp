#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace gufo::models::hy4_flash {

/// A `_hy4top_` reference: n_ctx, k, rows; then tokens, mask, reference
/// log-probabilities, and the reference IDs.
struct TopkReference {
  int n_ctx{0};
  int k{0};
  int rows{0};
  std::vector<std::int32_t> tokens;
  std::vector<std::uint8_t> mask;
  std::vector<float> reference_log_probs;
  std::vector<std::int32_t> ids;
};

/// One masked position's Gufo probabilities, tails, and KL from the reference.
struct KldPosition {
  int top_token{0};
  bool top1{false};
  double kl{0.0};
  double reference_tail{0.0};
  double gufo_tail{0.0};
  std::vector<double> gufo_log_probs;
};

bool LoadTopkReference(const std::filesystem::path& path, int expected_k,
                       TopkReference* reference, std::string* error);

/// Full-vocabulary log-softmax, the K reference IDs' Gufo log-probabilities,
/// both remaining masses, and the top-k KL from the reference to Gufo.
bool ScorePosition(const float* logits, int vocab,
                   const float* reference_log_probs,
                   const std::int32_t* reference_ids, int k,
                   KldPosition* position, std::string* error);

void WriteKldPosition(std::ostream& out, int row, int pos,
                      const KldPosition& position);

}  // namespace gufo::models::hy4_flash