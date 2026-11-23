#include "image_prep.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace gufo::models::hy4_flash {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

// Python round(): round-half-to-even on the binary double value.
int BankerRound(double value) {
  const double floored = std::floor(value);
  const double fraction = value - floored;
  if (fraction > 0.5) {
    return static_cast<int>(floored) + 1;
  }
  if (fraction < 0.5) {
    return static_cast<int>(floored);
  }
  const int low = static_cast<int>(floored);
  return (low % 2 == 0) ? low : low + 1;
}

}  // namespace

int VisionFactor(const VisionGeometry& geometry) {
  return geometry.patch * geometry.merge;
}

int VisionRows(int width, int height, const VisionGeometry& geometry) {
  const int factor = VisionFactor(geometry);
  if (factor <= 0 || width <= 0 || height <= 0 || width % factor != 0 ||
      height % factor != 0 ||
      static_cast<std::int64_t>(width) * height > geometry.max_pixels) {
    return 0;
  }
  return (height / factor) * (width / factor + 1);
}

bool SmartResize(int width, int height, const VisionGeometry& geometry,
                 int* out_width, int* out_height, std::string* error) {
  if (out_width == nullptr || out_height == nullptr) {
    SetError(error, "image resize output is missing");
    return false;
  }
  *out_width = 0;
  *out_height = 0;
  if (width <= 0 || height <= 0) {
    SetError(error, "image dimensions must be positive");
    return false;
  }
  const int factor = VisionFactor(geometry);
  if (factor <= 0) {
    SetError(error, "vision geometry has an invalid factor");
    return false;
  }
  const std::int64_t long_side = std::max(width, height);
  const std::int64_t short_side = std::max(1, std::min(width, height));
  if (long_side > 200 * short_side) {
    SetError(error, "the image aspect ratio is more than 200:1");
    return false;
  }
  std::int64_t height_bar = std::max<std::int64_t>(
      factor, static_cast<std::int64_t>(
                  BankerRound(static_cast<double>(height) / factor)) *
                  factor);
  std::int64_t width_bar = std::max<std::int64_t>(
      factor, static_cast<std::int64_t>(
                  BankerRound(static_cast<double>(width) / factor)) *
                  factor);
  if (height_bar * width_bar > geometry.max_pixels) {
    const double beta =
        std::sqrt(static_cast<double>(height) * width / geometry.max_pixels);
    height_bar = std::max<std::int64_t>(
        factor, static_cast<std::int64_t>(
                    std::floor(static_cast<double>(height) / beta / factor)) *
                    factor);
    width_bar = std::max<std::int64_t>(
        factor, static_cast<std::int64_t>(
                    std::floor(static_cast<double>(width) / beta / factor)) *
                    factor);
  } else if (height_bar * width_bar < geometry.min_pixels) {
    const double beta = std::sqrt(static_cast<double>(geometry.min_pixels) /
                                  (static_cast<double>(height) * width));
    height_bar = static_cast<std::int64_t>(
                     std::ceil(static_cast<double>(height) * beta / factor)) *
                 factor;
    width_bar = static_cast<std::int64_t>(
                    std::ceil(static_cast<double>(width) * beta / factor)) *
                factor;
  }
  *out_width = static_cast<int>(width_bar);
  *out_height = static_cast<int>(height_bar);
  return true;
}

bool PrepareImage(std::span<const std::uint8_t> bytes,
                  const VisionGeometry& geometry, gufo::core::Image* image,
                  std::string* error) {
  if (image == nullptr) {
    SetError(error, "image output is missing");
    return false;
  }
  try {
    *image = gufo::core::DecodeImage(
        bytes, {.composite_over_white = true, .apply_webp_exif = true});
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
  const int source_width = static_cast<int>(image->width);
  const int source_height = static_cast<int>(image->height);
  int target_width = 0;
  int target_height = 0;
  if (!SmartResize(source_width, source_height, geometry, &target_width,
                   &target_height, error)) {
    return false;
  }
  if (target_width != source_width || target_height != source_height) {
    std::vector<std::uint8_t> resized;
    if (!gufo::core::ResizeRgb8(image->pixels, source_width, source_height,
                                target_width, target_height, &resized)) {
      SetError(error, "image resize failed");
      return false;
    }
    image->width = static_cast<std::uint32_t>(target_width);
    image->height = static_cast<std::uint32_t>(target_height);
    image->pixels = std::move(resized);
  }
  if (VisionRows(target_width, target_height, geometry) <= 0) {
    SetError(error, "image dimensions do not fit the vision geometry");
    return false;
  }
  return true;
}

}  // namespace gufo::models::hy4_flash