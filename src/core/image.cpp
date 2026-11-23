#include "src/core/image.hpp"

#include <curl/curl.h>
#include <gufo/png_decode.h>
#include <jpeglib.h>
#include <webp/decode.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <csetjmp>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/core/base64.hpp"
#include "src/core/platform/socket.hpp"

namespace gufo::core {
namespace {

void ValidateDimensions(std::uint32_t width, std::uint32_t height) {
  if (width == 0 || height == 0 ||
      std::uint64_t{width} * height > kMaxImagePixels) {
    throw std::invalid_argument(
        "image dimensions exceed the decoded pixel limit");
  }
}

struct Filter {
  int begin{0};
  std::vector<std::int32_t> weights;
};

// Pillow's ImagingResample.c: Lanczos-3, 22-bit fixed-point coefficients.
std::vector<Filter> LanczosFilters(int input, int output) {
  std::vector<Filter> result(output);
  const double scale = static_cast<double>(input) / output;
  const double filter_scale = std::max(scale, 1.0);
  const double support = filter_scale * 3;
  for (int dst = 0; dst < output; ++dst) {
    const double center = (dst + 0.5) * scale;
    const int start = std::max(0, static_cast<int>(center - support + 0.5));
    const int end = std::min(input, static_cast<int>(center + support + 0.5));
    auto& f = result[dst];
    f.begin = start;
    std::vector<double> values;
    double sum = 0;
    for (int i = start; i < end; ++i) {
      const double x = (i - center + 0.5) / filter_scale;
      const auto sinc = [](double z) {
        return z == 0 ? 1.0
                      : std::sin(z * std::numbers::pi) / (z * std::numbers::pi);
      };
      const double w = (x >= -3 && x < 3) ? sinc(x) * sinc(x / 3) : 0;
      values.push_back(w);
      sum += w;
    }
    for (double value : values) {
      f.weights.push_back(static_cast<std::int32_t>(value / sum * (1 << 22) +
                                                    (value < 0 ? -0.5 : 0.5)));
    }
  }
  return result;
}

struct JpegError {
  jpeg_error_mgr manager{};
  std::jmp_buf jump;
};

struct JpegDecoder {
  jpeg_decompress_struct decoder{};
  JpegError error;
  Image image;
  std::vector<std::uint8_t> cmyk_row;
  bool created{false};
  ~JpegDecoder() {
    if (created)
      jpeg_destroy_decompress(&decoder);
  }
};

Image DecodeJpeg(std::span<const std::uint8_t> bytes) {
  // Keep every modified C++ object on the heap, outside libjpeg's longjmp.
  auto state = std::make_unique<JpegDecoder>();
  state->decoder.err = jpeg_std_error(&state->error.manager);
  state->error.manager.error_exit = [](j_common_ptr common) {
    std::longjmp(reinterpret_cast<JpegError*>(common->err)->jump, 1);
  };
  state->error.manager.output_message = [](j_common_ptr) {};
  if (setjmp(state->error.jump) != 0) {
    throw std::invalid_argument("invalid JPEG image");
  }
  jpeg_create_decompress(&state->decoder);
  state->created = true;
  jpeg_mem_src(&state->decoder, bytes.data(), bytes.size());
  if (jpeg_read_header(&state->decoder, TRUE) != JPEG_HEADER_OK) {
    throw std::invalid_argument("invalid JPEG header");
  }
  ValidateDimensions(state->decoder.image_width, state->decoder.image_height);
  const bool cmyk = state->decoder.jpeg_color_space == JCS_CMYK ||
                    state->decoder.jpeg_color_space == JCS_YCCK;
  state->decoder.out_color_space = cmyk ? JCS_CMYK : JCS_RGB;
  jpeg_start_decompress(&state->decoder);
  state->image.width = state->decoder.output_width;
  state->image.height = state->decoder.output_height;
  state->image.pixels.resize(std::size_t{state->image.width} *
                             state->image.height * 3);
  if (cmyk)
    state->cmyk_row.resize(std::size_t{state->image.width} * 4);
  while (state->decoder.output_scanline < state->decoder.output_height) {
    auto* rgb =
        state->image.pixels.data() +
        std::size_t{state->decoder.output_scanline} * state->image.width * 3;
    JSAMPROW row = cmyk ? state->cmyk_row.data() : rgb;
    if (jpeg_read_scanlines(&state->decoder, &row, 1) != 1) {
      throw std::invalid_argument("truncated JPEG image");
    }
    if (cmyk) {
      // Pillow decodes JPEG CMYK with inverted channels, then converts to RGB.
      for (std::size_t x = 0; x < state->image.width; ++x) {
        for (std::size_t c = 0; c < 3; ++c) {
          const unsigned product =
              unsigned{row[x * 4 + c]} * row[x * 4 + 3] + 128;
          rgb[x * 3 + c] =
              static_cast<std::uint8_t>((product + (product >> 8)) >> 8);
        }
      }
    }
  }
  jpeg_finish_decompress(&state->decoder);
  if (state->error.manager.num_warnings != 0)
    throw std::invalid_argument("corrupt or truncated JPEG image");
  return std::move(state->image);
}

void CheckPng(GufoPngStatus status) {
  if (status == GUFO_PNG_INVALID)
    throw std::invalid_argument("invalid PNG image");
  if (status != GUFO_PNG_OK)
    throw std::runtime_error("PNG decoder failed");
}

struct PngDecoderClose {
  void operator()(GufoPngDecoder* decoder) const { gufo_png_close(decoder); }
};

Image DecodePng(std::span<const std::uint8_t> bytes) {
  GufoPngDecoder* opened = nullptr;
  GufoPngDimensions size{};
  const GufoPngStatus status =
      gufo_png_open(bytes.data(), bytes.size(), &opened, &size);
  const std::unique_ptr<GufoPngDecoder, PngDecoderClose> decoder(opened);
  CheckPng(status);
  ValidateDimensions(size.width, size.height);
  Image image{size.width, size.height, {}};
  image.pixels.resize(std::size_t{image.width} * image.height * 3);
  CheckPng(gufo_png_decode_rgb8(decoder.get(), image.pixels.data(),
                                image.pixels.size()));
  return image;
}

struct RgbaImage {
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::vector<std::uint8_t> pixels;
};

RgbaImage DecodePngRgba(std::span<const std::uint8_t> bytes) {
  GufoPngDecoder* opened = nullptr;
  GufoPngDimensions size{};
  const GufoPngStatus status =
      gufo_png_open(bytes.data(), bytes.size(), &opened, &size);
  const std::unique_ptr<GufoPngDecoder, PngDecoderClose> decoder(opened);
  CheckPng(status);
  ValidateDimensions(size.width, size.height);
  RgbaImage image{size.width, size.height, {}};
  image.pixels.resize(std::size_t{image.width} * image.height * 4);
  CheckPng(gufo_png_decode_rgba8(decoder.get(), image.pixels.data(),
                                 image.pixels.size()));
  return image;
}

// Pillow's paste over an opaque white background: (rgb*a + 255*(255-a) + 127)
// / 255 for each channel. Orientation only moves pixels, so blending first
// gives the same bytes as rotating RGBA first and saves a rotated RGBA buffer.
Image CompositeOverWhite(const RgbaImage& rgba) {
  Image image{rgba.width, rgba.height, {}};
  image.pixels.resize(std::size_t{image.width} * image.height * 3);
  for (std::size_t i = 0; i < std::size_t{image.width} * image.height; ++i) {
    const unsigned alpha = rgba.pixels[i * 4 + 3];
    const unsigned inverse = 255 - alpha;
    for (int channel = 0; channel < 3; ++channel) {
      const unsigned value = rgba.pixels[i * 4 + channel];
      image.pixels[i * 3 + channel] = static_cast<std::uint8_t>(
          (value * alpha + 255 * inverse + 127) / 255);
    }
  }
  return image;
}

// Bounds-checked TIFF orientation; shared by JPEG APP1 and PNG eXIf.
unsigned TiffOrientation(std::span<const std::uint8_t> data) {
  if (data.size() < 8)
    return 1;
  const bool little = data[0] == 'I' && data[1] == 'I';
  if (!little && !(data[0] == 'M' && data[1] == 'M'))
    return 1;
  const auto read = [&](std::size_t pos, std::size_t size) -> std::uint32_t {
    if (pos > data.size() || size > data.size() - pos)
      return 0;
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < size; ++i) {
      value |= std::uint32_t{data[pos + i]}
               << (8 * (little ? i : size - 1 - i));
    }
    return value;
  };
  if (read(2, 2) != 42)
    return 1;
  const std::size_t ifd = read(4, 4);
  if (ifd > data.size() || data.size() - ifd < 2)
    return 1;
  const std::size_t count = read(ifd, 2);
  if (count > (data.size() - ifd - 2) / 12)
    return 1;
  for (std::size_t i = 0; i < count; ++i) {
    const auto p = ifd + 2 + i * 12;
    if (read(p, 2) == 0x112 && read(p + 2, 2) == 3 && read(p + 4, 4) == 1) {
      const unsigned value = read(p + 8, 2);
      return value >= 1 && value <= 8 ? value : 1;
    }
  }
  return 1;
}

unsigned JpegOrientation(std::span<const std::uint8_t> bytes) {
  for (std::size_t offset = 2; offset < bytes.size();) {
    if (bytes[offset] != 0xff)
      break;
    // JPEG permits repeated FF fill bytes before a marker.
    while (offset < bytes.size() && bytes[offset] == 0xff)
      ++offset;
    if (offset == bytes.size())
      break;
    const auto marker = bytes[offset++];
    if (marker == 0xda || marker == 0xd9)
      break;
    if (marker == 0x01 || marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd7))
      continue;
    if (bytes.size() - offset < 2)
      break;
    const std::size_t length =
        (unsigned{bytes[offset]} << 8) | bytes[offset + 1];
    if (length < 2 || length > bytes.size() - offset)
      break;
    const auto data = bytes.subspan(offset + 2, length - 2);
    if (marker == 0xe1 && data.size() >= 14 &&
        std::memcmp(data.data(), "Exif\0\0", 6) == 0) {
      return TiffOrientation(data.subspan(6));
    }
    offset += length;
  }
  return 1;
}

unsigned PngOrientation(std::span<const std::uint8_t> bytes) {
  for (std::size_t offset = 8; offset + 12 <= bytes.size();) {
    const std::size_t length = (std::uint32_t{bytes[offset]} << 24) |
                               (std::uint32_t{bytes[offset + 1]} << 16) |
                               (std::uint32_t{bytes[offset + 2]} << 8) |
                               bytes[offset + 3];
    if (length > bytes.size() - offset - 12)
      break;
    if (std::memcmp(bytes.data() + offset + 4, "eXIf", 4) == 0) {
      return TiffOrientation(bytes.subspan(offset + 8, length));
    }
    offset += length + 12;
  }
  return 1;
}

Image Orient(Image image, unsigned orientation) {
  if (orientation == 1)
    return image;
  const bool transpose = orientation >= 5;
  Image result{transpose ? image.height : image.width,
               transpose ? image.width : image.height,
               std::vector<std::uint8_t>(image.pixels.size())};
  for (std::uint32_t y = 0; y < result.height; ++y) {
    for (std::uint32_t x = 0; x < result.width; ++x) {
      std::uint32_t sx = x;
      std::uint32_t sy = y;
      switch (orientation) {
        case 2:
          sx = image.width - 1 - x;
          break;
        case 3:
          sx = image.width - 1 - x;
          sy = image.height - 1 - y;
          break;
        case 4:
          sy = image.height - 1 - y;
          break;
        case 5:
          sx = y;
          sy = x;
          break;
        case 6:
          sx = y;
          sy = image.height - 1 - x;
          break;
        case 7:
          sx = image.width - 1 - y;
          sy = image.height - 1 - x;
          break;
        case 8:
          sx = image.width - 1 - y;
          sy = x;
          break;
        default:
          break;
      }
      std::copy_n(
          image.pixels.data() + (std::size_t{sy} * image.width + sx) * 3, 3,
          result.pixels.data() + (std::size_t{y} * result.width + x) * 3);
    }
  }
  return result;
}

// WebP EXIF is a RIFF chunk. Some writers keep the JPEG-style Exif\0\0 prefix;
// others store the TIFF header directly. Strip the prefix only when present.
unsigned WebpOrientation(std::span<const std::uint8_t> bytes) {
  std::size_t offset = 12;
  while (offset + 8 <= bytes.size()) {
    const std::uint32_t size = std::uint32_t{bytes[offset + 4]} |
                               (std::uint32_t{bytes[offset + 5]} << 8) |
                               (std::uint32_t{bytes[offset + 6]} << 16) |
                               (std::uint32_t{bytes[offset + 7]} << 24);
    if (size > bytes.size() - offset - 8)
      break;
    if (std::memcmp(bytes.data() + offset, "EXIF", 4) == 0) {
      auto chunk = bytes.subspan(offset + 8, size);
      if (chunk.size() >= 6 && std::memcmp(chunk.data(), "Exif\0\0", 6) == 0)
        chunk = chunk.subspan(6);
      return TiffOrientation(chunk);
    }
    offset += 8 + static_cast<std::size_t>(size) + (size & 1U);
  }
  return 1;
}

std::vector<std::uint8_t> DecodeBase64(std::string_view input) {
  if (input.empty() || input.size() % 4 != 0 ||
      input.size() > ((kMaxEncodedImageBytes + 2) / 3) * 4) {
    throw std::invalid_argument("invalid or oversized base64 image");
  }
  std::vector<std::uint8_t> output;
  switch (core::DecodeBase64(input, {.max_decoded = kMaxEncodedImageBytes},
                             output)) {
    case core::Base64Status::kOk:
      return output;
    case core::Base64Status::kPadding:
      throw std::invalid_argument("invalid base64 image padding");
    case core::Base64Status::kNonCanonical:
      throw std::invalid_argument("noncanonical base64 image");
    case core::Base64Status::kTooLong:
      throw std::invalid_argument("image exceeds encoded byte limit");
    case core::Base64Status::kLength:
    case core::Base64Status::kAfterPadding:
    case core::Base64Status::kCharacter:
      break;
  }
  throw std::invalid_argument("invalid base64 image character");
}

// WebP, lossy or lossless; like PNG, alpha is dropped rather than blended.
Image DecodeWebp(std::span<const std::uint8_t> bytes) {
  int width = 0;
  int height = 0;
  if (WebPGetInfo(bytes.data(), bytes.size(), &width, &height) == 0)
    throw std::invalid_argument("invalid WebP image");
  ValidateDimensions(static_cast<std::uint32_t>(width),
                     static_cast<std::uint32_t>(height));
  Image image{static_cast<std::uint32_t>(width),
              static_cast<std::uint32_t>(height),
              {}};
  image.pixels.resize(std::size_t{image.width} * image.height * 3);
  if (WebPDecodeRGBInto(bytes.data(), bytes.size(), image.pixels.data(),
                        image.pixels.size(),
                        static_cast<int>(image.width) * 3) == nullptr)
    throw std::invalid_argument("invalid WebP image");
  return image;
}

RgbaImage DecodeWebpRgba(std::span<const std::uint8_t> bytes) {
  int width = 0;
  int height = 0;
  if (WebPGetInfo(bytes.data(), bytes.size(), &width, &height) == 0)
    throw std::invalid_argument("invalid WebP image");
  ValidateDimensions(static_cast<std::uint32_t>(width),
                     static_cast<std::uint32_t>(height));
  RgbaImage image{static_cast<std::uint32_t>(width),
                  static_cast<std::uint32_t>(height),
                  {}};
  image.pixels.resize(std::size_t{image.width} * image.height * 4);
  if (WebPDecodeRGBAInto(bytes.data(), bytes.size(), image.pixels.data(),
                         image.pixels.size(),
                         static_cast<int>(image.width) * 4) == nullptr)
    throw std::invalid_argument("invalid WebP image");
  return image;
}

// PNG colour types 3, 4 and 6 carry alpha, or a palette that expands to alpha.
// Types 0 and 2 keep their gray or RGB colour-key values.
bool PngCompositesOverWhite(std::span<const std::uint8_t> bytes) {
  if (bytes.size() < 26)
    return false;
  const std::uint8_t color_type = bytes[25];
  return color_type == 3 || color_type == 4 || color_type == 6;
}

}  // namespace

bool ResizeRgb8(const std::vector<std::uint8_t>& source, int source_width,
                int source_height, int width, int height,
                std::vector<std::uint8_t>* output) {
  if (output == nullptr || source_width <= 0 || source_height <= 0 ||
      width <= 0 || height <= 0 ||
      source.size() != static_cast<std::size_t>(source_width) *
                           static_cast<std::size_t>(source_height) * 3) {
    return false;
  }
  if (width == source_width && height == source_height) {
    *output = source;
    return true;
  }
  const auto horizontal = LanczosFilters(source_width, width);
  const auto vertical = LanczosFilters(source_height, height);
  std::vector<std::uint8_t> temp(static_cast<std::size_t>(width) *
                                 source_height * 3);
  for (int y = 0; y < source_height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        std::int64_t sum = 1 << 21;
        const auto& f = horizontal[x];
        for (std::size_t k = 0; k < f.weights.size(); ++k) {
          sum += static_cast<std::int64_t>(f.weights[k]) *
                 source[(static_cast<std::size_t>(y) * source_width + f.begin +
                         k) *
                            3 +
                        c];
        }
        temp[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
            static_cast<std::uint8_t>(
                std::clamp<std::int64_t>(sum >> 22, 0, 255));
      }
    }
  }
  output->resize(static_cast<std::size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        std::int64_t sum = 1 << 21;
        const auto& f = vertical[y];
        for (std::size_t k = 0; k < f.weights.size(); ++k) {
          sum += static_cast<std::int64_t>(f.weights[k]) *
                 temp[((f.begin + k) * width + x) * 3 + c];
        }
        (*output)[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
            static_cast<std::uint8_t>(
                std::clamp<std::int64_t>(sum >> 22, 0, 255));
      }
    }
  }
  return true;
}

Image DecodeImage(std::span<const std::uint8_t> bytes) {
  return DecodeImage(bytes, {});
}

Image DecodeImage(std::span<const std::uint8_t> bytes,
                  const ImageDecodeOptions& options) {
  if (bytes.empty() || bytes.size() > kMaxEncodedImageBytes) {
    throw std::invalid_argument("image exceeds encoded byte limit or is empty");
  }
  constexpr std::array<std::uint8_t, 8> kPngSignature{0x89, 'P',  'N',  'G',
                                                      '\r', '\n', 0x1a, '\n'};
  if (bytes.size() >= kPngSignature.size() &&
      std::memcmp(bytes.data(), kPngSignature.data(), kPngSignature.size()) ==
          0) {
    if (options.composite_over_white && PngCompositesOverWhite(bytes)) {
      return Orient(CompositeOverWhite(DecodePngRgba(bytes)),
                    PngOrientation(bytes));
    }
    return Orient(DecodePng(bytes), PngOrientation(bytes));
  }
  if (bytes.size() >= 2 && bytes[0] == 0xff && bytes[1] == 0xd8) {
    return Orient(DecodeJpeg(bytes), JpegOrientation(bytes));
  }
  if (bytes.size() >= 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0 &&
      std::memcmp(bytes.data() + 8, "WEBP", 4) == 0) {
    if (options.composite_over_white) {
      const unsigned orientation =
          options.apply_webp_exif ? WebpOrientation(bytes) : 1;
      return Orient(CompositeOverWhite(DecodeWebpRgba(bytes)), orientation);
    }
    Image image = DecodeWebp(bytes);
    if (options.apply_webp_exif)
      return Orient(std::move(image), WebpOrientation(bytes));
    return image;
  }
  throw std::invalid_argument("image must be PNG, JPEG or WebP");
}

std::vector<std::uint8_t> ReadImageFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::invalid_argument("cannot open image file");
  const auto length = input.tellg();
  if (length <= 0 ||
      length > static_cast<std::streamoff>(kMaxEncodedImageBytes)) {
    throw std::invalid_argument(
        "image file exceeds encoded byte limit or is empty");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char*>(bytes.data()), length)) {
    throw std::invalid_argument("cannot read image file");
  }
  return bytes;
}

enum class MediaKind { kImage, kAudio, kVideo };

static std::vector<std::uint8_t> ReadMediaUrl(std::string_view url,
                                              ImageReadBudget& budget,
                                              MediaKind media_kind);

std::vector<std::uint8_t> ReadImageUrl(std::string_view url,
                                       ImageReadBudget& budget) {
  return ReadMediaUrl(url, budget, MediaKind::kImage);
}
std::vector<std::uint8_t> ReadAudioUrl(std::string_view url,
                                       ImageReadBudget& budget) {
  if (url.starts_with("data:") || url.starts_with("https://") ||
      url.starts_with("http://") || url.starts_with("file:"))
    return ReadMediaUrl(url, budget, MediaKind::kAudio);
  // Raw base64 has no scheme. The strict decoder rejects paths and bad padding.
  return ReadMediaUrl("data:audio/wav;base64," + std::string(url), budget,
                      MediaKind::kAudio);
}
std::vector<std::uint8_t> ReadVideoUrl(std::string_view url,
                                       ImageReadBudget& budget) {
  if (url.starts_with("data:") || url.starts_with("https://") ||
      url.starts_with("http://") || url.starts_with("file:"))
    return ReadMediaUrl(url, budget, MediaKind::kVideo);
  return ReadMediaUrl("data:video/mp4;base64," + std::string(url), budget,
                      MediaKind::kVideo);
}
std::vector<std::uint8_t> ReadImageUrl(std::string_view url) {
  ImageReadBudget budget;
  return ReadImageUrl(url, budget);
}

bool IsPublicImageAddress(std::span<const std::uint8_t> address) noexcept {
  if (address.size() == 4) {
    const auto a = address[0], b = address[1], c = address[2];
    return a != 0 && a != 10 && a != 127 && a < 224 &&
           !(a == 100 && b >= 64 && b <= 127) && !(a == 169 && b == 254) &&
           !(a == 172 && b >= 16 && b <= 31) &&
           !(a == 192 && (b == 168 || (b == 0 && (c == 0 || c == 2)) ||
                          (b == 88 && c == 99))) &&
           !(a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100))) &&
           !(a == 203 && b == 0 && c == 113);
  }
  if (address.size() != 16 || (address[0] & 0xe0) != 0x20)
    return false;
  // Only native global unicast; no mapped IPv4, NAT64, Teredo or 6to4
  // tunnelling, which could otherwise reach an embedded private IPv4 address.
  if (address[0] == 0x20 &&
      ((address[1] == 0x01 &&
        (address[2] < 2 || (address[2] == 0x0d && address[3] == 0xb8))) ||
       address[1] == 0x02))
    return false;
  return !(address[0] == 0x3f && address[1] == 0xff &&
           (address[2] & 0xf0) == 0);
}

static std::vector<std::uint8_t> ReadMediaUrl(std::string_view url,
                                              ImageReadBudget& budget,
                                              MediaKind media_kind) {
  if (!budget.accepts_images)
    throw std::invalid_argument("model does not support image input");
  const auto remaining_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          budget.deadline - std::chrono::steady_clock::now())
          .count();
  if (budget.remaining_bytes == 0 || remaining_ms <= 0)
    throw std::invalid_argument("request image byte or time budget exceeded");
  if (url.starts_with("data:")) {
    const auto comma = url.find(',');
    if (comma == std::string_view::npos || comma > 256)
      throw std::invalid_argument("image data URL has no data");
    // A media type, parameters and ";base64", case-insensitive (RFC 2397).
    // Clients differ in case, "image/jpg" and parameters such as a file
    // name; the decoder tells the formats apart by their bytes anyway.
    std::string header(url.substr(5, comma - 5));
    std::ranges::transform(header, header.begin(), [](unsigned char c) {
      return static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    });
    if (!header.ends_with(";base64"))
      throw std::invalid_argument("image data URL must be base64-encoded");
    const std::string media = header.substr(0, header.find(';'));
    if (media_kind == MediaKind::kVideo
            ? (media != "video/mp4" && media != "video/webm" &&
               media != "video/x-matroska" && media != "video/quicktime")
        : media_kind == MediaKind::kAudio
            ? (media != "audio/wav" && media != "audio/x-wav" &&
               media != "audio/mpeg" && media != "audio/mp3" &&
               media != "audio/flac" && media != "audio/ogg")
            : (media != "image/png" && media != "image/jpeg" &&
               media != "image/jpg" && media != "image/webp"))
      throw std::invalid_argument(
          media_kind != MediaKind::kImage
              ? "unsupported media data URL type"
              : "image data URL type \"" + media +
                    "\" is not supported; use PNG, JPEG or WebP");
    const auto encoded = url.substr(comma + 1);
    // Check the decoded size before allocating, including base64 padding.
    const auto padding = encoded.ends_with("==")  ? 2U
                         : encoded.ends_with("=") ? 1U
                                                  : 0U;
    if (encoded.size() / 4 * 3 < padding ||
        encoded.size() / 4 * 3 - padding > budget.remaining_bytes)
      throw std::invalid_argument("request image byte budget exceeded");
    auto bytes = DecodeBase64(encoded);
    budget.remaining_bytes -= bytes.size();
    return bytes;
  }
  if (!url.starts_with("https://") || url.size() > 8192) {
    throw std::invalid_argument(
        "image URL must use HTTPS or a base64 data URL");
  }
  const std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(
      curl_easy_init(), curl_easy_cleanup);
  if (!curl)
    throw std::runtime_error("cannot initialize image download");
  std::vector<std::uint8_t> bytes;
  const std::string address(url);
  const auto set = [&](CURLoption option, auto value) {
    if (curl_easy_setopt(curl.get(), option, value) != CURLE_OK)
      throw std::runtime_error("cannot configure image download");
  };
  set(CURLOPT_URL, address.c_str());
  set(CURLOPT_PROTOCOLS_STR, "https");
  set(CURLOPT_REDIR_PROTOCOLS_STR, "https");
  set(CURLOPT_DISALLOW_USERNAME_IN_URL, 1L);
  // A proxy would hide the destination address from our socket callback.
  set(CURLOPT_PROXY, "");
  set(CURLOPT_FOLLOWLOCATION, 1L);
  set(CURLOPT_MAXREDIRS, 3L);
  set(CURLOPT_TIMEOUT_MS, static_cast<long>(remaining_ms));
  set(CURLOPT_CONNECTTIMEOUT_MS,
      std::min(5000L, static_cast<long>(remaining_ms)));
  set(CURLOPT_NOSIGNAL, 1L);
  set(CURLOPT_FAILONERROR, 1L);
  set(CURLOPT_MAXFILESIZE_LARGE,
      static_cast<curl_off_t>(budget.remaining_bytes));
  set(
      CURLOPT_OPENSOCKETFUNCTION,
      +[](void*, curlsocktype purpose,
          curl_sockaddr* endpoint) -> curl_socket_t {
        if (purpose != CURLSOCKTYPE_IPCXN)
          return CURL_SOCKET_BAD;
        std::span<const std::uint8_t> ip;
        if (endpoint->family == AF_INET &&
            endpoint->addrlen >= sizeof(sockaddr_in)) {
          const auto* v4 =
              reinterpret_cast<const sockaddr_in*>(&endpoint->addr);
          ip = {reinterpret_cast<const std::uint8_t*>(&v4->sin_addr), 4};
        } else if (endpoint->family == AF_INET6 &&
                   endpoint->addrlen >= sizeof(sockaddr_in6)) {
          const auto* v6 =
              reinterpret_cast<const sockaddr_in6*>(&endpoint->addr);
          if (v6->sin6_scope_id != 0)
            return CURL_SOCKET_BAD;
          ip = {reinterpret_cast<const std::uint8_t*>(&v6->sin6_addr), 16};
        }
        if (!IsPublicImageAddress(ip))
          return CURL_SOCKET_BAD;
        return ::socket(endpoint->family,
                        endpoint->socktype | platform::kSocketCloexec,
                        endpoint->protocol);
      });
  struct Download {
    std::vector<std::uint8_t>& bytes;
    ImageReadBudget& budget;
  } download{bytes, budget};
  set(CURLOPT_WRITEDATA, &download);
  set(
      CURLOPT_WRITEFUNCTION,
      +[](char* data, std::size_t size, std::size_t count,
          void* opaque) -> std::size_t {
        auto& state = *static_cast<Download*>(opaque);
        auto& output = state.bytes;
        if (size != 0 && count > state.budget.remaining_bytes / size) {
          return 0;
        }
        const std::size_t length = size * count;
        state.budget.remaining_bytes -= length;
        try {
          output.insert(output.end(), data, data + length);
        } catch (...) {
          return 0;
        }
        return length;
      });
  if (curl_easy_perform(curl.get()) != CURLE_OK || bytes.empty()) {
    throw std::invalid_argument(
        "cannot download image within size/time limits");
  }
  return bytes;
}

}  // namespace gufo::core
