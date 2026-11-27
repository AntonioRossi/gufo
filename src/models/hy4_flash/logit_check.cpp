#include "logit_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <utility>

namespace gufo::models::hy4_flash {
namespace {

constexpr double kProbabilityEpsilon = 1e-5;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool ReadExact(std::ifstream* input, void* data, std::size_t bytes) {
  input->read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
  return static_cast<bool>(*input);
}

bool CheckedMultiply(std::uint64_t left, std::uint64_t right,
                     std::uint64_t* output) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *output = left * right;
  return true;
}

bool CheckedAdd(std::uint64_t left, std::uint64_t right,
                std::uint64_t* output) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  *output = left + right;
  return true;
}

}  // namespace

bool LoadTopkReference(const std::filesystem::path& path, int expected_k,
                       TopkReference* reference, std::string* error) {
  if (reference == nullptr) {
    SetError(error, "top-k reference output is missing");
    return false;
  }
  *reference = {};
  if (expected_k <= 0) {
    SetError(error, "expected k must be positive");
    return false;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    SetError(error, path.string() + ": cannot open");
    return false;
  }
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (size < 0) {
    SetError(error, path.string() + ": cannot get the file size");
    return false;
  }
  const std::uintmax_t file_size = static_cast<std::uintmax_t>(size);
  char magic[8];
  std::int32_t header[3] = {0, 0, 0};
  if (!ReadExact(&input, magic, sizeof magic) ||
      std::memcmp(magic, "_hy4top_", sizeof magic) != 0 ||
      !ReadExact(&input, header, sizeof header)) {
    SetError(error, path.string() + ": not a top-k reference");
    return false;
  }
  const int n_ctx = header[0];
  const int k = header[1];
  const int rows = header[2];
  if (n_ctx < 3 || rows != 1 || k != expected_k) {
    SetError(error, path.string() + ": invalid reference geometry");
    return false;
  }
  std::uint64_t token_count = 0;
  std::uint64_t score_count = 0;
  std::uint64_t token_bytes = 0;
  std::uint64_t mask_bytes = 0;
  std::uint64_t rlp_bytes = 0;
  std::uint64_t id_bytes = 0;
  if (!CheckedMultiply(static_cast<std::uint64_t>(rows),
                       static_cast<std::uint64_t>(n_ctx), &token_count) ||
      !CheckedMultiply(token_count, static_cast<std::uint64_t>(k),
                       &score_count) ||
      !CheckedMultiply(token_count, 4, &token_bytes) ||
      !CheckedMultiply(static_cast<std::uint64_t>(rows),
                       static_cast<std::uint64_t>(n_ctx - 1), &mask_bytes) ||
      !CheckedMultiply(score_count, 4, &rlp_bytes) ||
      !CheckedMultiply(score_count, 4, &id_bytes)) {
    SetError(error, path.string() + ": reference is too large");
    return false;
  }
  std::uint64_t total = 20;
  const std::uint64_t parts[] = {token_bytes, mask_bytes, rlp_bytes, id_bytes};
  for (const std::uint64_t part : parts) {
    if (!CheckedAdd(total, part, &total)) {
      SetError(error, path.string() + ": reference is too large");
      return false;
    }
  }
  if (total > std::numeric_limits<std::size_t>::max()) {
    SetError(error, path.string() + ": reference is too large");
    return false;
  }
  if (file_size != static_cast<std::uintmax_t>(total)) {
    SetError(error, path.string() + ": unexpected reference length");
    return false;
  }
  reference->tokens.resize(static_cast<std::size_t>(token_count));
  reference->mask.resize(static_cast<std::size_t>(mask_bytes));
  reference->reference_log_probs.resize(static_cast<std::size_t>(score_count));
  reference->ids.resize(static_cast<std::size_t>(score_count));
  if (!ReadExact(&input, reference->tokens.data(),
                 reference->tokens.size() * sizeof(std::int32_t)) ||
      !ReadExact(&input, reference->mask.data(), reference->mask.size()) ||
      !ReadExact(&input, reference->reference_log_probs.data(),
                 reference->reference_log_probs.size() * sizeof(float)) ||
      !ReadExact(&input, reference->ids.data(),
                 reference->ids.size() * sizeof(std::int32_t))) {
    SetError(error, path.string() + ": cut short");
    return false;
  }
  reference->n_ctx = n_ctx;
  reference->k = k;
  reference->rows = rows;
  return true;
}

bool ScorePosition(const float* logits, int vocab,
                   const float* reference_log_probs,
                   const std::int32_t* reference_ids, int k,
                   KldPosition* position, std::string* error) {
  if (logits == nullptr || reference_log_probs == nullptr ||
      reference_ids == nullptr || position == nullptr || vocab <= 0 || k <= 0) {
    SetError(error, "invalid logit check input");
    return false;
  }
  const auto maximum_it = std::max_element(logits, logits + vocab);
  const float maximum = *maximum_it;
  if (!std::isfinite(maximum)) {
    SetError(error, "non-finite logits");
    return false;
  }
  double sum = 0.0;
  for (int i = 0; i < vocab; ++i) {
    if (!std::isfinite(logits[i])) {
      SetError(error, "non-finite logits");
      return false;
    }
    sum +=
        std::exp(static_cast<double>(logits[i]) - static_cast<double>(maximum));
  }
  if (!(sum > 0.0)) {
    SetError(error, "invalid logit sum");
    return false;
  }
  const double log_sum = std::log(sum);
  position->gufo_log_probs.assign(static_cast<std::size_t>(k), 0.0);
  double reference_mass = 0.0;
  double gufo_mass = 0.0;
  double kl = 0.0;
  for (int j = 0; j < k; ++j) {
    const std::int32_t id = reference_ids[j];
    if (id < 0 || id >= vocab) {
      SetError(error, "reference ID outside the vocabulary");
      return false;
    }
    if (!std::isfinite(reference_log_probs[j])) {
      SetError(error, "non-finite reference log-probability");
      return false;
    }
    if (reference_log_probs[j] > 0.0F) {
      SetError(error, "positive reference log-probability");
      return false;
    }
    for (int earlier = 0; earlier < j; ++earlier) {
      if (reference_ids[earlier] == id) {
        SetError(error, "duplicate reference ID");
        return false;
      }
    }
    if (j > 0 && reference_log_probs[j] > reference_log_probs[0]) {
      SetError(error, "the first reference ID is not the highest");
      return false;
    }
    const double reference_log = static_cast<double>(reference_log_probs[j]);
    const double gufo_log =
        (static_cast<double>(logits[id]) - maximum) - log_sum;
    if (!std::isfinite(gufo_log)) {
      SetError(error, "non-finite Gufo log-probability");
      return false;
    }
    const double reference = std::exp(reference_log);
    position->gufo_log_probs[static_cast<std::size_t>(j)] = gufo_log;
    kl += reference * (reference_log - gufo_log);
    reference_mass += reference;
    gufo_mass += std::exp(gufo_log);
  }
  if (reference_mass > 1.0 + kProbabilityEpsilon) {
    SetError(error, "reference probability mass above one");
    return false;
  }
  const double reference_tail = std::max(1e-9, 1.0 - reference_mass);
  const double gufo_tail = std::max(1e-9, 1.0 - gufo_mass);
  position->kl =
      kl + reference_tail * (std::log(reference_tail) - std::log(gufo_tail));
  if (!std::isfinite(position->kl) || !std::isfinite(reference_tail) ||
      !std::isfinite(gufo_tail)) {
    SetError(error, "non-finite KL result");
    return false;
  }
  position->reference_tail = reference_tail;
  position->gufo_tail = gufo_tail;
  position->top_token = static_cast<int>(maximum_it - logits);
  position->top1 = position->top_token == reference_ids[0];
  return true;
}

void WriteKldPosition(std::ostream& out, int row, int pos,
                      const KldPosition& position) {
  out << row << ' ' << pos << ' ' << std::setprecision(9) << position.kl << ' '
      << (position.top1 ? 1 : 0) << ' ' << position.top_token << ' '
      << position.reference_tail << ' ' << position.gufo_tail;
  for (const double value : position.gufo_log_probs) {
    out << ' ' << value;
  }
  out << '\n';
}

}  // namespace gufo::models::hy4_flash