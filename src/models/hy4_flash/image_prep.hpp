#pragma once

#include <cstdint>
#include <gufo/ext/image.hpp>
#include <span>
#include <string>

namespace gufo::models::hy4_flash {

/// The vision fields the image preparation needs from the loaded pack.
struct VisionGeometry {
  int patch{16};
  int merge{2};
  int min_pixels{4096};
  int max_pixels{4194304};
};

int VisionFactor(const VisionGeometry& geometry);
/// The image tokens: (height / factor) * (width / factor + 1).
int VisionRows(int width, int height, const VisionGeometry& geometry);
/// Maya's smart_resize: aspect limit, banker's rounding, pixel bounds.
bool SmartResize(int width, int height, const VisionGeometry& geometry,
                 int* out_width, int* out_height, std::string* error);
/// Decode with white alpha and WebP EXIF, then smart-resize to the geometry.
bool PrepareImage(std::span<const std::uint8_t> bytes,
                  const VisionGeometry& geometry, gufo::core::Image* image,
                  std::string* error);

}  // namespace gufo::models::hy4_flash