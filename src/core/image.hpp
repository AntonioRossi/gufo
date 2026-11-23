#ifndef GUFO_CORE_IMAGE_HPP_
#define GUFO_CORE_IMAGE_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core {

inline constexpr std::size_t kMaxEncodedImageBytes = 20 * 1024 * 1024;
inline constexpr std::size_t kMaxImagePixels = 32 * 1024 * 1024;

struct Image {
  std::uint32_t width{0};
  std::uint32_t height{0};
  /// Packed, unpremultiplied RGB8, in display orientation.
  std::vector<std::uint8_t> pixels;
};

/// Decode choices that do not apply to the historical callers.
struct ImageDecodeOptions {
  /// Composite PNG colour types 3, 4 and 6 and transparent WebP over white.
  /// PNG colour types 0 and 2 keep their gray or RGB colour-key values. The
  /// default drops alpha without compositing, as the historical callers expect.
  bool composite_over_white{false};
  /// Apply the WebP EXIF orientation. PNG and JPEG always apply theirs.
  bool apply_webp_exif{false};
};

/// PNG, JPEG and WebP. Decoding checks dimensions before allocating pixels.
[[nodiscard]] Image DecodeImage(std::span<const std::uint8_t> bytes);
/// The same formats with explicit alpha and WebP EXIF choices.
[[nodiscard]] Image DecodeImage(std::span<const std::uint8_t> bytes,
                                const ImageDecodeOptions& options);
/// Two-pass Lanczos resize of packed RGB8 with Pillow parity.
[[nodiscard]] bool ResizeRgb8(const std::vector<std::uint8_t>& source,
                              int source_width, int source_height, int width,
                              int height, std::vector<std::uint8_t>* output);
[[nodiscard]] std::vector<std::uint8_t> ReadImageFile(
    const std::filesystem::path& path);
/// Shared across all messages of one HTTP request.
struct ImageReadBudget {
  /// False when the loaded model takes no image input; reading is refused
  /// before any data URL is decoded or any URL is fetched.
  bool accepts_images{true};
  std::size_t remaining_bytes{kMaxEncodedImageBytes};
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::now() + std::chrono::seconds(15)};
};

/// Network-order IPv4/IPv6 address; excludes local and special-use ranges.
[[nodiscard]] bool IsPublicImageAddress(
    std::span<const std::uint8_t> address) noexcept;
/// Base64 data URLs or public HTTPS destinations. DNS results and every
/// redirect's actual socket address are checked before connecting.
[[nodiscard]] std::vector<std::uint8_t> ReadImageUrl(std::string_view url,
                                                     ImageReadBudget& budget);
[[nodiscard]] std::vector<std::uint8_t> ReadImageUrl(std::string_view url);
/// Audio uses the same bounded reader and destination checks. Supports raw
/// base64, audio data URLs and public HTTPS. The decoder checks the bytes.
[[nodiscard]] std::vector<std::uint8_t> ReadAudioUrl(std::string_view url,
                                                     ImageReadBudget& budget);

/// Video uses the same shared byte/deadline and public HTTPS protections.
[[nodiscard]] std::vector<std::uint8_t> ReadVideoUrl(std::string_view url,
                                                     ImageReadBudget& budget);

}  // namespace gufo::core
#endif
