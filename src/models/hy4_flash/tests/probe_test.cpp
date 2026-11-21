#include "../probe.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../provider.hpp"
#include "gguf_fixture.hpp"

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

}  // namespace

int main(int argc, char** argv) {
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
      Require("real model metadata", ProbeHy4Flash(model));
      std::string error;
      const bool verified = VerifyHy4FlashModel(model, &error);
      Require("real model strict", verified, error);
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

    const bool pack_written = hy4_test::WritePack(dir, pair, &write_error);
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