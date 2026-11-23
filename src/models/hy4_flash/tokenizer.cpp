#include "tokenizer.hpp"

#include <unicode/parseerr.h>
#include <unicode/regex.h>
#include <unicode/stringpiece.h>
#include <unicode/unistr.h>
#include <unicode/utypes.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gufo::models::hy4_flash {
namespace {

constexpr const char* kStageOne = R"(\p{N}{1,3})";
constexpr const char* kStageTwo = "[一-龥぀-ゟ゠-ヿ]+";
constexpr const char* kStageThree =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|"
    "[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
    " ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::optional<std::string> ReadFile(const std::filesystem::path& path,
                                    std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    if (error != nullptr) {
      *error = path.string() + ": cannot open";
    }
    return std::nullopt;
  }
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

icu::UnicodeString Utf8(std::string_view text) {
  return icu::UnicodeString::fromUTF8(
      icu::StringPiece(text.data(), static_cast<int32_t>(text.size())));
}

std::string ToUtf8(const icu::UnicodeString& text) {
  std::string result;
  text.toUTF8String(result);
  return result;
}

std::string Utf8Encode(std::uint32_t codepoint) {
  std::string out;
  if (codepoint <= 0x7f) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ff) {
    out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else if (codepoint <= 0xffff) {
    out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  }
  return out;
}

std::vector<std::uint32_t> Utf8ToCodepoints(std::string_view text) {
  std::vector<std::uint32_t> out;
  for (std::size_t i = 0; i < text.size();) {
    const auto first = static_cast<std::uint8_t>(text[i]);
    std::uint32_t codepoint = first;
    std::size_t width = 1;
    if ((first & 0xe0) == 0xc0) {
      codepoint = first & 0x1f;
      width = 2;
    } else if ((first & 0xf0) == 0xe0) {
      codepoint = first & 0x0f;
      width = 3;
    } else if ((first & 0xf8) == 0xf0) {
      codepoint = first & 0x07;
      width = 4;
    }
    if (i + width > text.size()) {
      width = 1;
      codepoint = first;
    } else {
      for (std::size_t j = 1; j < width; ++j) {
        codepoint =
            (codepoint << 6) | (static_cast<std::uint8_t>(text[i + j]) & 0x3f);
      }
    }
    out.push_back(codepoint);
    i += width;
  }
  return out;
}

std::array<std::uint32_t, 256> MakeByteToUnicode() {
  std::array<std::uint32_t, 256> mapping{};
  std::array<bool, 256> direct{};
  std::vector<int> bytes;
  for (int value = 0x21; value <= 0x7e; ++value) {
    bytes.push_back(value);
    direct[static_cast<std::size_t>(value)] = true;
  }
  for (int value = 0xa1; value <= 0xac; ++value) {
    bytes.push_back(value);
    direct[static_cast<std::size_t>(value)] = true;
  }
  for (int value = 0xae; value <= 0xff; ++value) {
    bytes.push_back(value);
    direct[static_cast<std::size_t>(value)] = true;
  }
  std::vector<int> codepoints = bytes;
  int next = 0;
  for (int value = 0; value < 256; ++value) {
    if (!direct[static_cast<std::size_t>(value)]) {
      bytes.push_back(value);
      codepoints.push_back(256 + next);
      ++next;
    }
  }
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    mapping[static_cast<std::size_t>(bytes[i])] =
        static_cast<std::uint32_t>(codepoints[i]);
  }
  return mapping;
}

std::string EscapeRegex(std::string_view literal) {
  std::string out;
  for (const char value : literal) {
    if (std::strchr("\\.^$|?*+()[]{}", value) != nullptr) {
      out.push_back('\\');
    }
    out.push_back(value);
  }
  return out;
}

std::unique_ptr<icu::RegexPattern> CompilePattern(std::string_view pattern) {
  UErrorCode status = U_ZERO_ERROR;
  UParseError parse_error{};
  std::unique_ptr<icu::RegexPattern> compiled(
      icu::RegexPattern::compile(Utf8(pattern), 0, parse_error, status));
  if (U_FAILURE(status) || !compiled) {
    throw std::runtime_error(
        std::string("cannot compile the tokenizer pattern: ") +
        u_errorName(status));
  }
  return compiled;
}

}  // namespace

struct Tokenizer::Impl {
  std::vector<std::string> tokens;
  std::unordered_map<std::string, int> ids;
  std::unordered_map<std::string, int> ranks;
  std::vector<int> types;
  std::unordered_map<std::string, int> always;
  std::unordered_map<int, std::string> aliases;
  int eos = -1;
  std::array<std::uint32_t, 256> byte_to_unicode{};
  std::unordered_map<std::uint32_t, std::uint8_t> unicode_to_byte;
  std::vector<std::unique_ptr<icu::RegexPattern>> stages;
  std::unique_ptr<icu::RegexPattern> always_pattern;

  std::vector<int> Bpe(std::string_view piece) const {
    std::vector<std::string> parts;
    parts.reserve(piece.size());
    for (const unsigned char value : piece) {
      parts.push_back(Utf8Encode(byte_to_unicode[value]));
    }
    while (parts.size() > 1) {
      std::size_t best = parts.size();
      int best_rank = std::numeric_limits<int>::max();
      for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
        const auto found = ranks.find(parts[i] + " " + parts[i + 1]);
        if (found != ranks.end() && found->second < best_rank) {
          best = i;
          best_rank = found->second;
        }
      }
      if (best == parts.size()) {
        break;
      }
      parts[best] += parts[best + 1];
      parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best) + 1);
    }
    std::vector<int> out;
    out.reserve(parts.size());
    for (const auto& part : parts) {
      const auto found = ids.find(part);
      if (found == ids.end()) {
        throw std::runtime_error("BPE produced a token outside the vocabulary");
      }
      out.push_back(found->second);
    }
    return out;
  }

  std::vector<int> EncodePlain(std::string_view text) const {
    std::vector<std::string> pieces{std::string(text)};
    for (const auto& pattern : stages) {
      std::vector<std::string> next;
      for (const auto& piece : pieces) {
        const icu::UnicodeString input = Utf8(piece);
        UErrorCode status = U_ZERO_ERROR;
        std::unique_ptr<icu::RegexMatcher> matcher(
            pattern->matcher(input, status));
        if (U_FAILURE(status) || !matcher) {
          throw std::runtime_error("cannot create the tokenizer matcher");
        }
        int32_t position = 0;
        while (matcher->find(status) && U_SUCCESS(status)) {
          const int32_t start = matcher->start(status);
          const int32_t end = matcher->end(status);
          if (U_FAILURE(status)) {
            throw std::runtime_error("tokenizer match position failed");
          }
          if (start > position) {
            next.push_back(ToUtf8(input.tempSubStringBetween(position, start)));
          }
          if (end > start) {
            next.push_back(ToUtf8(input.tempSubStringBetween(start, end)));
          }
          position = end;
        }
        if (U_FAILURE(status)) {
          throw std::runtime_error("tokenizer match failed");
        }
        if (position < input.length()) {
          next.push_back(ToUtf8(input.tempSubString(position)));
        }
      }
      pieces = std::move(next);
    }
    std::vector<int> out;
    for (const auto& piece : pieces) {
      const auto encoded = Bpe(piece);
      out.insert(out.end(), encoded.begin(), encoded.end());
    }
    return out;
  }

  std::vector<int> Encode(std::string_view text) const {
    if (!always_pattern) {
      return EncodePlain(text);
    }
    const icu::UnicodeString input = Utf8(text);
    UErrorCode status = U_ZERO_ERROR;
    std::unique_ptr<icu::RegexMatcher> matcher(
        always_pattern->matcher(input, status));
    if (U_FAILURE(status) || !matcher) {
      throw std::runtime_error("cannot create the special-token matcher");
    }
    std::vector<int> out;
    int32_t position = 0;
    while (matcher->find(status) && U_SUCCESS(status)) {
      const int32_t start = matcher->start(status);
      const int32_t end = matcher->end(status);
      if (U_FAILURE(status)) {
        throw std::runtime_error("special-token match position failed");
      }
      if (start > position) {
        const auto plain =
            EncodePlain(ToUtf8(input.tempSubStringBetween(position, start)));
        out.insert(out.end(), plain.begin(), plain.end());
      }
      const auto found = always.find(ToUtf8(matcher->group(status)));
      if (found == always.end()) {
        throw std::runtime_error("unknown special token");
      }
      out.push_back(found->second);
      position = end;
    }
    if (U_FAILURE(status)) {
      throw std::runtime_error("special-token match failed");
    }
    if (position < input.length()) {
      const auto plain = EncodePlain(ToUtf8(input.tempSubString(position)));
      out.insert(out.end(), plain.begin(), plain.end());
    }
    return out;
  }

  std::string DecodeId(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= tokens.size()) {
      throw std::runtime_error("token id outside the vocabulary");
    }
    const auto alias = aliases.find(id);
    if (alias != aliases.end()) {
      return alias->second;
    }
    if (static_cast<std::size_t>(id) < types.size() &&
        (types[static_cast<std::size_t>(id)] == 3 ||
         types[static_cast<std::size_t>(id)] == 4)) {
      return tokens[static_cast<std::size_t>(id)];
    }
    std::string out;
    for (const std::uint32_t codepoint :
         Utf8ToCodepoints(tokens[static_cast<std::size_t>(id)])) {
      const auto found = unicode_to_byte.find(codepoint);
      if (found == unicode_to_byte.end()) {
        throw std::runtime_error(
            "token contains a character outside the byte map");
      }
      out.push_back(static_cast<char>(found->second));
    }
    return out;
  }
};

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::~Tokenizer() = default;

std::optional<Tokenizer> Tokenizer::Load(const std::filesystem::path& directory,
                                         std::string* error) {
  try {
    auto impl = std::make_unique<Impl>();

    auto vocab_text = ReadFile(directory / "vocab.json", error);
    if (!vocab_text) {
      return std::nullopt;
    }
    const auto vocab = gufo::json::parse(*vocab_text);
    if (vocab.type() != gufo::json::Value::Type::kObject) {
      throw std::runtime_error("vocab.json is not an object");
    }
    const std::size_t count = vocab.members().size();
    if (count == 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("vocab.json has an invalid entry count");
    }
    impl->tokens.resize(count);
    std::vector<bool> seen(count, false);
    for (const auto& [token, value] : vocab.members()) {
      const std::size_t id = value.as_size(count);
      if (id >= count || seen[id] ||
          !impl->ids.emplace(token, static_cast<int>(id)).second) {
        throw std::runtime_error("vocab.json has a duplicate or invalid id");
      }
      seen[id] = true;
      impl->tokens[id] = token;
    }

    auto types_text = ReadFile(directory / "token_type.json", error);
    if (!types_text) {
      return std::nullopt;
    }
    const auto type_json = gufo::json::parse(*types_text);
    if (type_json.type() != gufo::json::Value::Type::kArray ||
        type_json.items().size() != impl->tokens.size()) {
      throw std::runtime_error("token_type.json does not match vocab.json");
    }
    impl->types.reserve(type_json.items().size());
    for (const auto& value : type_json.items()) {
      impl->types.push_back(static_cast<int>(value.as_size()));
    }

    const auto merges_text = ReadFile(directory / "merges.txt", error);
    if (!merges_text) {
      return std::nullopt;
    }
    std::istringstream merges(*merges_text);
    int rank = 0;
    for (std::string line; std::getline(merges, line);) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.empty()) {
        continue;
      }
      const std::size_t space = line.find(' ');
      if (space == std::string::npos) {
        throw std::runtime_error("merges.txt has a line without a pair");
      }
      impl->ranks[line.substr(0, space) + " " + line.substr(space + 1)] =
          rank++;
    }

    const auto meta_text = ReadFile(directory / "tokenizer.json", error);
    if (!meta_text) {
      return std::nullopt;
    }
    const auto meta = gufo::json::parse(*meta_text);
    if (meta.type() != gufo::json::Value::Type::kObject ||
        meta.member_str("pre") != "hy_v4") {
      throw std::runtime_error("tokenizer.json does not declare hy_v4");
    }
    if (const auto* special = meta.find("special_ids");
        special != nullptr &&
        special->type() == gufo::json::Value::Type::kObject) {
      impl->eos = static_cast<int>(
          special->member_size("eos", static_cast<std::size_t>(-1)));
    }
    if (const auto* aliases = meta.find("aliases");
        aliases != nullptr &&
        aliases->type() == gufo::json::Value::Type::kObject) {
      for (const auto& [key, value] : aliases->members()) {
        impl->aliases[std::stoi(key)] = value.str();
      }
    }

    for (std::size_t id = 0; id < impl->tokens.size(); ++id) {
      if (impl->types[id] == 4 && !impl->tokens[id].empty()) {
        impl->always.emplace(impl->tokens[id], static_cast<int>(id));
      }
    }
    if (!impl->always.empty()) {
      std::vector<std::string> literals;
      literals.reserve(impl->always.size());
      for (const auto& [text, id] : impl->always) {
        (void)id;
        literals.push_back(text);
      }
      std::sort(literals.begin(), literals.end(),
                [](const std::string& left, const std::string& right) {
                  return left.size() > right.size();
                });
      icu::UnicodeString pattern;
      for (std::size_t i = 0; i < literals.size(); ++i) {
        if (i != 0) {
          pattern.append(static_cast<UChar32>('|'));
        }
        pattern.append(Utf8(EscapeRegex(literals[i])));
      }
      impl->always_pattern = CompilePattern(ToUtf8(pattern));
    }

    impl->stages.push_back(CompilePattern(kStageOne));
    impl->stages.push_back(CompilePattern(kStageTwo));
    impl->stages.push_back(CompilePattern(kStageThree));

    impl->byte_to_unicode = MakeByteToUnicode();
    for (std::size_t value = 0; value < impl->byte_to_unicode.size(); ++value) {
      impl->unicode_to_byte[impl->byte_to_unicode[value]] =
          static_cast<std::uint8_t>(value);
    }

    return Tokenizer(std::move(impl));
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return std::nullopt;
  }
}

std::vector<int> Tokenizer::Encode(std::string_view text) const {
  return impl_->Encode(text);
}

std::string Tokenizer::Decode(std::span<const int> ids) const {
  std::string bytes;
  for (const int id : ids) {
    bytes += impl_->DecodeId(id);
  }
  return ToUtf8(Utf8(bytes));
}

int Tokenizer::eos() const {
  return impl_->eos;
}

}  // namespace gufo::models::hy4_flash