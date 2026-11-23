#include "../probe.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "../provider.hpp"
#include "../tokenizer.hpp"
#include "gguf_fixture.hpp"

#ifdef HY4_IMAGE_CHECKS
#include "../image_prep.hpp"
#endif
#include "../logit_check.hpp"

void hy4_flash_register(gufo::ext::Registry& registry);

namespace {

struct CheckFailure {};

void Require(const char* name, bool condition, std::string_view detail = {}) {
  if (condition) {
    return;
  }
  if (detail.empty()) {
    std::fprintf(stderr, "FAIL %s\n", name);
  } else {
    std::fprintf(stderr, "FAIL %s: %.*s\n", name,
                 static_cast<int>(detail.size()), detail.data());
  }
  throw CheckFailure{};
}

struct DirectoryGuard {
  std::filesystem::path path;

  ~DirectoryGuard() {
    if (!path.empty()) {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  }
};

std::unique_ptr<gufo::core::GgufReader> Open(const hy4_test::Fixture& fixture) {
  static std::vector<std::uint8_t> bytes;
  bytes = hy4_test::BuildSingle(fixture);
  std::string error;
  auto reader = gufo::core::GgufReader::OpenMemory(
      bytes.data(), bytes.size(), &error, {.allow_unsized_tensors = true});
  if (!reader) {
    std::fprintf(stderr, "fixture did not parse: %s\n", error.c_str());
  }
  return reader;
}

void ExpectProbe(const char* name, const hy4_test::Fixture& fixture,
                 bool expected) {
  const auto reader = Open(fixture);
  Require(name, reader && gufo::models::hy4_flash::ProbeHy4FlashGguf(*reader) ==
                              expected);
}

std::vector<std::string> ReadLines(const std::filesystem::path& path) {
  std::vector<std::string> lines;
  std::ifstream input(path);
  for (std::string line; std::getline(input, line);) {
    lines.push_back(std::move(line));
  }
  return lines;
}

bool WriteLines(const std::filesystem::path& path,
                const std::vector<std::string>& lines) {
  std::ofstream output(path);
  if (!output) {
    return false;
  }
  for (const auto& line : lines) {
    output << line << '\n';
  }
  output.close();
  return static_cast<bool>(output);
}

std::vector<std::string> Split(const std::string& line) {
  std::istringstream input(line);
  std::vector<std::string> tokens;
  for (std::string token; input >> token;) {
    tokens.push_back(std::move(token));
  }
  return tokens;
}

std::string Join(const std::vector<std::string>& tokens) {
  std::string line;
  for (const auto& token : tokens) {
    if (!line.empty()) {
      line.push_back(' ');
    }
    line += token;
  }
  return line;
}

std::filesystem::path MakeDirectory() {
  const char* tmp = std::getenv("TMPDIR");
  if (tmp == nullptr || *tmp == '\0') {
    std::fprintf(stderr, "TMPDIR is not set\n");
    return {};
  }
  const auto pattern =
      (std::filesystem::path(tmp) / "hy4_flash_test.XXXXXX").string();
  std::vector<char> buffer(pattern.begin(), pattern.end());
  buffer.push_back('\0');
  char* created = mkdtemp(buffer.data());
  if (created == nullptr) {
    std::fprintf(stderr, "mkdtemp failed for %s\n", pattern.c_str());
    return {};
  }
  return std::filesystem::path(created);
}

std::optional<std::string> ReadText(const std::filesystem::path& path,
                                    std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    if (error != nullptr) {
      *error = path.string() + ": cannot open";
    }
    return std::nullopt;
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::optional<std::vector<int>> ReadIds(const std::filesystem::path& path,
                                        std::string* error) {
  const auto text = ReadText(path, error);
  if (!text) {
    return std::nullopt;
  }
  std::vector<int> ids;
  std::size_t position = 0;
  while (position < text->size()) {
    while (position < text->size() &&
           std::isspace(static_cast<unsigned char>((*text)[position]))) {
      ++position;
    }
    if (position == text->size()) {
      break;
    }
    std::size_t end = position;
    while (end < text->size() &&
           !std::isspace(static_cast<unsigned char>((*text)[end])) &&
           (*text)[end] != ',') {
      ++end;
    }
    int id = 0;
    const auto result =
        std::from_chars(text->data() + position, text->data() + end, id);
    if (result.ec != std::errc{} || result.ptr != text->data() + end) {
      if (error != nullptr) {
        *error = path.string() + ": invalid ID field";
      }
      return std::nullopt;
    }
    ids.push_back(id);
    position = end;
    if (position < text->size() && (*text)[position] == ',') {
      ++position;
    }
  }
  return ids;
}

int RunStage1Tokenizer(int argc, char** argv) {
  std::filesystem::path tokenizer;
  std::filesystem::path short_prompt;
  std::filesystem::path short_ids;
  std::filesystem::path full_prompt;
  std::filesystem::path full_ids;
  int alias = -1;
  for (int i = 2; i < argc; ++i) {
    const std::string option = argv[i];
    const auto value = [&]() -> std::string {
      if (++i >= argc) {
        throw CheckFailure{};
      }
      return argv[i];
    };
    if (option == "--tokenizer") {
      tokenizer = value();
    } else if (option == "--short") {
      short_prompt = value();
    } else if (option == "--short-ids") {
      short_ids = value();
    } else if (option == "--full") {
      full_prompt = value();
    } else if (option == "--full-ids") {
      full_ids = value();
    } else if (option == "--alias") {
      alias = std::stoi(value());
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", option.c_str());
      return 2;
    }
  }
  Require("tokenizer directory", std::filesystem::is_directory(tokenizer));
  Require("short prompt", std::filesystem::is_regular_file(short_prompt));
  Require("short ids", std::filesystem::is_regular_file(short_ids));
  Require("full prompt", std::filesystem::is_regular_file(full_prompt));
  Require("full ids", std::filesystem::is_regular_file(full_ids));
  Require("alias argument", alias >= 0);

  std::string error;
  auto loaded = gufo::models::hy4_flash::Tokenizer::Load(tokenizer, &error);
  Require("tokenizer load", loaded.has_value(), error);

  std::string read_error;
  const auto short_text = ReadText(short_prompt, &read_error);
  Require("short prompt read", short_text.has_value(), read_error);
  Require("short prompt empty", !short_text->empty());
  const auto short_expected = ReadIds(short_ids, &read_error);
  Require("short ids parse", short_expected.has_value(), read_error);
  Require("short ids count", short_expected->size() == 10);
  const auto short_actual = loaded->Encode(*short_text);
  Require("short ids", short_actual == *short_expected,
          std::to_string(short_actual.size()) + " vs " +
              std::to_string(short_expected->size()));

  const auto full_text = ReadText(full_prompt, &read_error);
  Require("full prompt read", full_text.has_value(), read_error);
  Require("full prompt empty", !full_text->empty());
  const auto full_expected = ReadIds(full_ids, &read_error);
  Require("full ids parse", full_expected.has_value(), read_error);
  Require("full ids count", full_expected->size() == 524224);
  const auto full_actual = loaded->Encode(*full_text);
  Require("full ids", full_actual == *full_expected,
          std::to_string(full_actual.size()) + " vs " +
              std::to_string(full_expected->size()));

  const auto meta_text = ReadText(tokenizer / "tokenizer.json", &read_error);
  Require("tokenizer meta read", meta_text.has_value(), read_error);
  const auto meta = gufo::json::parse(*meta_text);
  const auto* aliases = meta.find("aliases");
  Require("aliases", aliases != nullptr &&
                         aliases->type() == gufo::json::Value::Type::kObject);
  const auto* alias_value = aliases->find(std::to_string(alias));
  Require("alias entry", alias_value != nullptr && alias_value->is_string());
  const std::string alias_expected = alias_value->str();
  const auto alias_actual = loaded->Decode(std::span<const int>(&alias, 1));
  Require("alias bytes", alias_actual == alias_expected,
          std::to_string(alias_actual.size()) + " vs " +
              std::to_string(alias_expected.size()));

  std::printf("Stage 1 tokenizer: short %zu, full %zu, alias %zu bytes\n",
              short_actual.size(), full_actual.size(), alias_actual.size());
  return 0;
}

int RunLogitCheck() {
  const auto dir = MakeDirectory();
  if (dir.empty()) {
    return 1;
  }
  DirectoryGuard guard{dir};
  const auto path = dir / "synthetic._hy4top_";
  constexpr int n_ctx = 4;
  constexpr int k = 2;
  constexpr int rows = 1;
  const std::int32_t tokens[n_ctx] = {1, 2, 3, 4};
  const std::uint8_t mask[n_ctx - 1] = {0, 1, 1};
  const float reference_log_probs[n_ctx * k] = {
      0.0F, 0.0F, 0.0F, 0.0F, std::log(0.5F), std::log(0.25F), 0.0F, 0.0F};
  const std::int32_t reference_ids[n_ctx * k] = {0, 0, 0, 0, 1, 2, 0, 0};
  {
    std::ofstream out(path, std::ios::binary);
    out.write("_hy4top_", 8);
    const std::int32_t header[3] = {n_ctx, k, rows};
    out.write(reinterpret_cast<const char*>(header), sizeof header);
    out.write(reinterpret_cast<const char*>(tokens), sizeof tokens);
    out.write(reinterpret_cast<const char*>(mask), sizeof mask);
    out.write(reinterpret_cast<const char*>(reference_log_probs),
              sizeof reference_log_probs);
    out.write(reinterpret_cast<const char*>(reference_ids),
              sizeof reference_ids);
    Require("synthetic reference write", static_cast<bool>(out));
  }

  using gufo::models::hy4_flash::KldPosition;
  using gufo::models::hy4_flash::LoadTopkReference;
  using gufo::models::hy4_flash::ScorePosition;
  using gufo::models::hy4_flash::TopkReference;
  using gufo::models::hy4_flash::WriteKldPosition;
  TopkReference reference;
  std::string error;
  Require("reference load", LoadTopkReference(path, 2, &reference, &error),
          error);
  Require("reference shape", reference.n_ctx == n_ctx && reference.k == k &&
                                 reference.rows == rows);

  TopkReference wrong_k;
  Require("wrong expected k", !LoadTopkReference(path, 64, &wrong_k, &error),
          error);

  const auto truncated = dir / "truncated._hy4top_";
  {
    std::ifstream input(path, std::ios::binary);
    std::ofstream output(truncated, std::ios::binary);
    char header[20];
    input.read(header, sizeof header);
    output.write(header, sizeof header);
    Require("truncated reference write", static_cast<bool>(output));
  }
  TopkReference short_reference;
  Require("truncated reference",
          !LoadTopkReference(truncated, 2, &short_reference, &error), error);

  const float same_logits[4] = {std::log(0.125F), std::log(0.5F),
                                std::log(0.25F), std::log(0.125F)};
  KldPosition same;
  Require("self score",
          ScorePosition(same_logits, 4,
                        reference.reference_log_probs.data() + 2 * k,
                        reference.ids.data() + 2 * k, k, &same, &error),
          error);
  Require("self KL", std::abs(same.kl) < 1e-12);

  const float other_logits[4] = {0.0F, 3.0F, 1.0F, 0.0F};
  KldPosition other;
  Require("other score",
          ScorePosition(other_logits, 4,
                        reference.reference_log_probs.data() + 2 * k,
                        reference.ids.data() + 2 * k, k, &other, &error),
          error);
  Require("known KL", std::abs(other.kl - 0.247990059) < 1e-6);
  Require("gufo log prob count", other.gufo_log_probs.size() == 2);
  Require("reference tail", std::abs(other.reference_tail - 0.25) < 1e-6);

  const float wrong_order_log_probs[2] = {-1.000004F, -1.0F};
  const std::int32_t ordered_ids[2] = {1, 2};
  KldPosition wrong_order;
  Require("wrong reference order rejected",
          !ScorePosition(other_logits, 4, wrong_order_log_probs, ordered_ids, 2,
                         &wrong_order, &error),
          error);

  const float positive_log_probs[2] = {0.000001F, -30.0F};
  KldPosition positive;
  Require("positive reference log-probability rejected",
          !ScorePosition(other_logits, 4, positive_log_probs, ordered_ids, 2,
                         &positive, &error),
          error);

  const float uniform_logits[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  const float offset_logits[4] = {1e20F, 1e20F, 1e20F, 1e20F};
  KldPosition uniform;
  KldPosition offset;
  Require("uniform score",
          ScorePosition(uniform_logits, 4,
                        reference.reference_log_probs.data() + 2 * k,
                        reference.ids.data() + 2 * k, k, &uniform, &error),
          error);
  Require("offset score",
          ScorePosition(offset_logits, 4,
                        reference.reference_log_probs.data() + 2 * k,
                        reference.ids.data() + 2 * k, k, &offset, &error),
          error);
  Require("offset KL", std::abs(uniform.kl - offset.kl) < 1e-6);

  const float duplicate_log_probs[2] = {std::log(0.5F), std::log(0.25F)};
  const std::int32_t duplicate_ids[2] = {1, 1};
  KldPosition duplicate;
  Require("duplicate ID rejected",
          !ScorePosition(other_logits, 4, duplicate_log_probs, duplicate_ids, 2,
                         &duplicate, &error),
          error);

  const float heavy_log_probs[2] = {std::log(0.75F), std::log(0.75F)};
  const std::int32_t heavy_ids[2] = {1, 2};
  KldPosition heavy;
  Require("mass above one rejected",
          !ScorePosition(other_logits, 4, heavy_log_probs, heavy_ids, 2, &heavy,
                         &error),
          error);

  const float infinite_logits[4] = {0.0F, INFINITY, 1.0F, 0.0F};
  KldPosition infinite;
  Require("non-finite logits rejected",
          !ScorePosition(infinite_logits, 4,
                         reference.reference_log_probs.data() + 2 * k,
                         reference.ids.data() + 2 * k, 2, &infinite, &error),
          error);

  std::ostringstream line;
  WriteKldPosition(line, 0, 2, other);
  Require("written line", line.str().rfind("0 2 ", 0) == 0);
  std::printf("Logit check: self KL 0, known KL %.9g\n", other.kl);
  return 0;
}

#ifdef HY4_IMAGE_CHECKS
int RunPrepareImage(int argc, char** argv) {
  std::filesystem::path image;
  std::filesystem::path rgb;
  std::filesystem::path size;
  for (int i = 2; i < argc; ++i) {
    const std::string option = argv[i];
    const auto value = [&]() -> std::string {
      if (++i >= argc) {
        throw CheckFailure{};
      }
      return argv[i];
    };
    if (option == "--image") {
      image = value();
    } else if (option == "--rgb") {
      rgb = value();
    } else if (option == "--size") {
      size = value();
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", option.c_str());
      return 2;
    }
  }
  Require("image path", std::filesystem::is_regular_file(image),
          image.string());
  Require("rgb path", !rgb.empty());
  Require("size path", !size.empty());

  const gufo::models::hy4_flash::VisionGeometry geometry;
  std::string error;
  const auto bytes = gufo::core::ReadImageFile(image);
  gufo::core::Image prepared;
  const bool prepared_ok =
      gufo::models::hy4_flash::PrepareImage(bytes, geometry, &prepared, &error);
  Require("image prepare", prepared_ok, error);

  std::ofstream rgb_file(rgb, std::ios::binary);
  rgb_file.write(reinterpret_cast<const char*>(prepared.pixels.data()),
                 static_cast<std::streamsize>(prepared.pixels.size()));
  rgb_file.close();
  Require("rgb write", static_cast<bool>(rgb_file), rgb.string());

  std::ofstream size_file(size);
  size_file << prepared.width << "x" << prepared.height;
  size_file.close();
  Require("size write", static_cast<bool>(size_file), size.string());

  const int rows = gufo::models::hy4_flash::VisionRows(
      static_cast<int>(prepared.width), static_cast<int>(prepared.height),
      geometry);
  std::printf("%ux%u %d\n", prepared.width, prepared.height, rows);
  return 0;
}

int RunResizeOnly(int argc, char** argv) {
  int width = 0;
  int height = 0;
  for (int i = 2; i < argc; ++i) {
    const std::string option = argv[i];
    const auto value = [&]() -> std::string {
      if (++i >= argc) {
        throw CheckFailure{};
      }
      return argv[i];
    };
    if (option == "--width") {
      width = std::stoi(value());
    } else if (option == "--height") {
      height = std::stoi(value());
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", option.c_str());
      return 2;
    }
  }
  Require("width", width > 0);
  Require("height", height > 0);

  const gufo::models::hy4_flash::VisionGeometry geometry;
  int target_width = 0;
  int target_height = 0;
  std::string error;
  const bool resized = gufo::models::hy4_flash::SmartResize(
      width, height, geometry, &target_width, &target_height, &error);
  Require("resize", resized, error);
  std::printf("%dx%d\n", target_width, target_height);
  return 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string_view(argv[1]) == "--stage1-tokenizer") {
    try {
      return RunStage1Tokenizer(argc, argv);
    } catch (const CheckFailure&) {
      return 1;
    } catch (const std::exception& exception) {
      std::fprintf(stderr, "FAIL %s\n", exception.what());
      return 1;
    }
  }

  if (argc > 1 && std::string_view(argv[1]) == "--logit-check") {
    try {
      return RunLogitCheck();
    } catch (const CheckFailure&) {
      return 1;
    } catch (const std::exception& exception) {
      std::fprintf(stderr, "FAIL %s\n", exception.what());
      return 1;
    }
  }

#ifdef HY4_IMAGE_CHECKS
  if (argc > 1 && std::string_view(argv[1]) == "--prepare-image") {
    try {
      return RunPrepareImage(argc, argv);
    } catch (const CheckFailure&) {
      return 1;
    } catch (const std::exception& exception) {
      std::fprintf(stderr, "FAIL %s\n", exception.what());
      return 1;
    }
  }
  if (argc > 1 && std::string_view(argv[1]) == "--resize-only") {
    try {
      return RunResizeOnly(argc, argv);
    } catch (const CheckFailure&) {
      return 1;
    } catch (const std::exception& exception) {
      std::fprintf(stderr, "FAIL %s\n", exception.what());
      return 1;
    }
  }
#endif

  using gufo::models::hy4_flash::ProbeHy4Flash;
  using gufo::models::hy4_flash::ProbeHy4FlashGguf;
  using gufo::models::hy4_flash::VerifyHy4FlashModel;

  std::string model;
  if (argc > 1) {
    model = argv[1];
  } else if (const char* environment = std::getenv("HY4_FLASH_MODEL")) {
    model = environment;
  }
  if (argc > 2) {
    std::fprintf(stderr, "unknown argument: %s\n", argv[2]);
    return 2;
  }

  try {
    if (!model.empty()) {
      std::string error;
      const bool verified = VerifyHy4FlashModel(model, &error);
      Require("real model strict", verified, error);
      Require("real model path probe", ProbeHy4Flash(model));
      return 0;
    }

    const auto dir = MakeDirectory();
    if (dir.empty()) {
      return 1;
    }
    DirectoryGuard guard{dir};

    const hy4_test::BuiltPair pair = hy4_test::BuildPair(hy4_test::Fixture{});
    std::string write_error;
    const bool shards_written = hy4_test::WritePair(dir, pair, &write_error);
    Require("fixture shards written", shards_written, write_error);
    const auto shard1 = dir / hy4_test::kShard1Name;
    const auto shard2 = dir / hy4_test::kShard2Name;

    std::string open_error;
    const auto reader = gufo::core::GgufReader::OpenFile(
        shard1, &open_error, {.allow_unsized_tensors = true});
    Require("fixture pair opens", reader != nullptr, open_error);
    gufo::ext::RegisterTextProvider({.id = "hy4-flash-nosized",
                                     .probe = &ProbeHy4Flash,
                                     .probe_gguf = &ProbeHy4FlashGguf,
                                     .accepts_unsized_tensors = false});
    Require("provider without the flag is skipped",
            gufo::ext::ProbeTextProvider(*reader) == nullptr);
    gufo::ext::Registry registry;
    hy4_flash_register(registry);
    const auto* provider = gufo::ext::ProbeTextProvider(*reader);
    Require("hy4-flash is selected", provider != nullptr &&
                                         provider->id == "hy4-flash" &&
                                         provider->accepts_unsized_tensors);

    Require("path probe accepts the pair without a pack",
            ProbeHy4Flash(shard1.string()));
    {
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
      Require("verifier rejects a missing pack",
              rejected && error.find("missing pack") != std::string::npos,
              error);
    }
    {
      const bool two_file_written = hy4_test::WritePack(
          dir, pair, {.include_companions = false}, &write_error);
      Require("two-file pack written", two_file_written, write_error);
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
      Require("verifier rejects a two-file pack",
              rejected && error == "missing vision companion", error);
    }

    ExpectProbe("standard trunk", hy4_test::Fixture{}, true);
    {
      hy4_test::Fixture fixture;
      fixture.general_type = "mtp";
      ExpectProbe("MTP sidecar", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.general_type = "drafter";
      ExpectProbe("drafter sidecar", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.architecture = "clip";
      ExpectProbe("vision sidecar", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.attention_type = 12;
      ExpectProbe("Q4_K attention", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.dense_ffn_type = 12;
      ExpectProbe("Q4_K dense feed-forward", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.expert_type = 12;
      ExpectProbe("Q4_K experts", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.shared_type = 12;
      ExpectProbe("Q4_K shared expert", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.include_engram = false;
      ExpectProbe("missing Engram", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.engram_type = 1;
      ExpectProbe("sized Engram", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.engram_width = 64;
      ExpectProbe("Engram width mismatch", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.truncate_engram = true;
      ExpectProbe("Engram outside its span", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.nextn_predict_layers = 1;
      ExpectProbe("MTP block in the set", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.block_count = 41;
      ExpectProbe("41 blocks", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.expert_count = 256;
      ExpectProbe("256 experts", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.dense_blocks = 0;
      ExpectProbe("no dense layer", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.ngram = 1;
      ExpectProbe("one-gram Engram", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.vocabularies = {1, 4};
      ExpectProbe("Engram vocabulary below two", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.offsets = {0, 5};
      ExpectProbe("Engram rows outside the table", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.multipliers = {1};
      ExpectProbe("Engram multiplier count", fixture, false);
    }
    {
      hy4_test::Fixture fixture;
      fixture.split_count = 1;
      ExpectProbe("not the two-shard pair", fixture, false);
    }

    const bool pack_written = hy4_test::WritePack(dir, pair, {}, &write_error);
    Require("fixture pack written", pack_written, write_error);
    const auto index_path = dir / "maya-hy4" / "tensors.txt";
    {
      std::string error;
      const bool accepted = VerifyHy4FlashModel(shard1.string(), &error);
      Require("verifier accepts the first shard", accepted, error);
    }
    {
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard2.string(), &error);
      Require("verifier rejects the second shard",
              rejected && error.find("the first shard was not opened") !=
                              std::string::npos,
              error);
    }

    const auto write_case = [&](const hy4_test::Fixture& fixture,
                                const hy4_test::PackOptions& options) {
      const auto built = hy4_test::BuildPair(fixture);
      std::string error;
      const bool pair_written = hy4_test::WritePair(dir, built, &error);
      Require("case pair written", pair_written, error);
      const bool pack_written =
          hy4_test::WritePack(dir, built, options, &error);
      Require("case pack written", pack_written, error);
    };
    const auto restore_positive = [&]() {
      std::string error;
      const bool pair_written = hy4_test::WritePair(dir, pair, &error);
      Require("positive pair restored", pair_written, error);
      const bool pack_written = hy4_test::WritePack(dir, pair, {}, &error);
      Require("positive pack restored", pack_written, error);
    };

    {
      hy4_test::Fixture fixture;
      fixture.vision_projector_type = "other";
      write_case(fixture, {});
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
      Require(
          "verifier rejects another projector",
          rejected && error.find("clip.projector_type") != std::string::npos,
          error);
    }
    {
      hy4_test::Fixture fixture;
      fixture.split_vision = true;
      write_case(fixture, {.split_vision = true});
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
      Require("verifier rejects a split vision",
              rejected && error.find("must be one file") != std::string::npos,
              error);
    }
    {
      hy4_test::Fixture fixture;
      fixture.companion_tensor_name = "output.weight";
      write_case(fixture, {});
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
      Require(
          "verifier rejects a duplicate tensor name",
          rejected && error.find("names a tensor twice") != std::string::npos,
          error);
    }
    {
      hy4_test::Fixture fixture;
      fixture.include_mtp = true;
      write_case(fixture, {.include_mtp = true});
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
      Require("verifier rejects an MTP companion",
              rejected && error.find("MTP companion") != std::string::npos,
              error);
    }

    {
      hy4_test::Fixture fixture;
      fixture.vision_dim = 0;
      fixture.vision_embedding_length = 0;
      write_case(fixture, {});
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
      Require("verifier rejects zero vision width",
              rejected && error.find("vision.dim") != std::string::npos, error);
    }
    {
      hy4_test::Fixture fixture;
      fixture.dfly_layers = 0;
      fixture.dfly_block_count = 0;
      write_case(fixture, {});
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
      Require("verifier rejects zero drafter layers",
              rejected && error.find("dfly_layers") != std::string::npos,
              error);
    }
    restore_positive();

    const auto geometry_path = dir / "maya-hy4" / "hy4.txt";
    const auto geometry = ReadLines(geometry_path);
    const auto reject_geometry = [&](const std::string& field,
                                     const std::string& replacement) {
      auto lines = geometry;
      lines.erase(std::remove_if(lines.begin(), lines.end(),
                                 [&](const std::string& line) {
                                   return line.rfind(field + " ", 0) == 0;
                                 }),
                  lines.end());
      if (!replacement.empty()) {
        lines.push_back(replacement);
      }
      const bool written = WriteLines(geometry_path, lines);
      Require("geometry index write", written, geometry_path.string());
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
      Require("verifier rejects a bad geometry field",
              rejected && error.find(field) != std::string::npos, error);
    };
    reject_geometry("vision.dim", "");
    reject_geometry("vision.dim", "vision.dim");
    reject_geometry("vision.dim", "vision.dim invalid");
    reject_geometry("vision.dim", "vision.dim 0");
    reject_geometry("vision.dim", "vision.dim 5");
    reject_geometry("dfly_layers", "");
    reject_geometry("dfly_layers", "dfly_layers");
    reject_geometry("dfly_layers", "dfly_layers invalid");
    reject_geometry("dfly_layers", "dfly_layers 0");
    reject_geometry("dfly_layers", "dfly_layers 2");
    const auto reject_repeated = [&](const std::string& line,
                                     const std::string& field) {
      auto lines = geometry;
      lines.push_back(line);
      const bool written = WriteLines(geometry_path, lines);
      Require("geometry index write", written, geometry_path.string());
      std::string error;
      const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
      Require("verifier rejects a repeated geometry field",
              rejected && error.find("repeated") != std::string::npos &&
                  error.find(field) != std::string::npos,
              error);
    };
    reject_repeated("vision.dim", "vision.dim");
    reject_repeated("vision.dim invalid", "vision.dim");
    reject_repeated("dfly_layers", "dfly_layers");
    reject_repeated("dfly_layers invalid", "dfly_layers");
    {
      const bool written = WriteLines(geometry_path, geometry);
      Require("geometry index restored", written, geometry_path.string());
    }

    {
      const auto lines = ReadLines(index_path);
      const auto engram =
          std::find_if(lines.begin(), lines.end(), [](const std::string& line) {
            return line.rfind("t per_layer_token_embd.weight ", 0) == 0;
          });
      Require("pack has the Engram entry", engram != lines.end());
      const auto at = static_cast<std::size_t>(engram - lines.begin());
      const auto original = *engram;
      const auto tokens = Split(original);
      const auto reject = [&](const char* name,
                              const std::vector<std::string>& text,
                              const std::string& expected) {
        const bool written = WriteLines(index_path, text);
        Require("pack index write", written, index_path.string());
        std::string error;
        const bool rejected = !VerifyHy4FlashModel(shard1.string(), &error);
        Require(name, rejected && error.find(expected) != std::string::npos,
                error);
      };

      auto wrong_offset = tokens;
      wrong_offset[4] = std::to_string(std::stoull(wrong_offset[4]) - 32);
      auto offset_lines = lines;
      offset_lines[at] = Join(wrong_offset);
      reject("pack rejects a wrong address", offset_lines, "address");

      auto wrong_payload = tokens;
      wrong_payload[5] = std::to_string(std::stoull(wrong_payload[5]) - 1);
      auto payload_lines = lines;
      payload_lines[at] = Join(wrong_payload);
      reject("pack rejects a wrong payload", payload_lines, "payload");

      const auto last_dimension = tokens.size() - 1;
      const auto expected =
          "tensors.txt:" + std::to_string(at + 1) + ": invalid dimension";

      const std::pair<const char*, const char*> dimension_cases[] = {
          {"pack rejects an invalid dimension", "+"},
          {"pack rejects an out-of-range dimension", "18446744073709551616"},
      };
      for (const auto& [name, token] : dimension_cases) {
        auto dimension = tokens;
        dimension[last_dimension] = token;
        auto dimension_lines = lines;
        dimension_lines[at] = Join(dimension);
        reject(name, dimension_lines, expected);
      }

      auto omitted = lines;
      omitted.erase(omitted.begin() + static_cast<std::ptrdiff_t>(at));
      reject("pack rejects an omitted tensor", omitted, "omits");

      auto duplicated = lines;
      duplicated.insert(duplicated.begin() + static_cast<std::ptrdiff_t>(at),
                        original);
      reject("pack rejects a duplicate tensor", duplicated, "twice");
    }

    std::puts("hy4_flash probe checks passed");
    return 0;
  } catch (const CheckFailure&) {
    return 1;
  }
}