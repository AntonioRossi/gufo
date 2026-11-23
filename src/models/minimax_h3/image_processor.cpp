// Keyframe canvas fitting and vision patch construction for MiniMax H3
// FL2VA conditioning (upstream issue #177). See image_processor.hpp for the
// reference semantics; local://h3-fl2va-reference.md Part 3 pins the exact
// PIL arithmetic.

#include "src/models/minimax_h3/image_processor.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace gufo::minimax_h3::image_processor {
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

bool PrepareKeyframe(const core::Image& image, int canvas_width,
                     int canvas_height, KeyframeFit fit,
                     std::vector<std::uint8_t>* rgb, std::string* error) {
  if (rgb == nullptr) {
    SetError(error, "MiniMax H3 keyframe output is missing");
    return false;
  }
  *rgb = {};
  if (canvas_width <= 0 || canvas_height <= 0 || canvas_width % 32 != 0 ||
      canvas_height % 32 != 0) {
    SetError(error,
             "MiniMax H3 keyframe canvas must be a positive multiple of 32");
    return false;
  }
  if (image.width == 0 || image.height == 0 ||
      image.pixels.size() !=
          static_cast<std::size_t>(image.width) * image.height * 3U) {
    SetError(error, "MiniMax H3 keyframe is not packed RGB8");
    return false;
  }
  const auto width = static_cast<int>(image.width);
  const auto height = static_cast<int>(image.height);
  if (width == canvas_width && height == canvas_height) {
    // Exact-size shortcut (before_encoder.py:136): no resampling.
    *rgb = image.pixels;
    return true;
  }
  if (fit == KeyframeFit::kStretch) {
    return core::ResizeRgb8(image.pixels, width, height, canvas_width,
                            canvas_height, rgb)
               ? true
               : (SetError(error, "MiniMax H3 keyframe stretch failed"), false);
  }
  // Aspect cover scale + center crop (before_encoder.py:150-158). The
  // arithmetic is explicit because VaeImageProcessor's crop mode differs by
  // +-1 px on many ratios.
  const double scale = std::max(static_cast<double>(canvas_width) / width,
                                static_cast<double>(canvas_height) / height);
  const int resized_width =
      std::max(canvas_width, BankerRound(static_cast<double>(width) * scale));
  const int resized_height =
      std::max(canvas_height, BankerRound(static_cast<double>(height) * scale));
  std::vector<std::uint8_t> scaled;
  if (resized_width == width && resized_height == height) {
    scaled = image.pixels;
  } else if (!core::ResizeRgb8(image.pixels, width, height, resized_width,
                               resized_height, &scaled)) {
    SetError(error, "MiniMax H3 keyframe cover resize failed");
    return false;
  }
  const int left = std::max(0, (resized_width - canvas_width) / 2);
  const int top = std::max(0, (resized_height - canvas_height) / 2);
  rgb->resize(static_cast<std::size_t>(canvas_width) * canvas_height * 3);
  for (int y = 0; y < canvas_height; ++y) {
    const std::uint8_t* source =
        scaled.data() +
        (static_cast<std::size_t>(top + y) * resized_width + left) * 3;
    std::uint8_t* destination =
        rgb->data() + static_cast<std::size_t>(y) * canvas_width * 3;
    std::copy_n(source, static_cast<std::size_t>(canvas_width) * 3,
                destination);
  }
  return true;
}

bool BuildKeyframeVision(const std::vector<std::uint8_t>& rgb, int width,
                         int height, KeyframeVision* output,
                         std::string* error) {
  if (output == nullptr) {
    SetError(error, "MiniMax H3 keyframe vision output is missing");
    return false;
  }
  *output = {};
  if (width <= 0 || height <= 0 || width % 32 != 0 || height % 32 != 0) {
    SetError(error,
             "MiniMax H3 keyframe size must be a positive multiple "
             "of 32 for vision patching");
    return false;
  }
  if (rgb.size() != static_cast<std::size_t>(width) * height * 3U) {
    SetError(error, "MiniMax H3 keyframe pixels do not cover the canvas");
    return false;
  }
  if (static_cast<std::int64_t>(width) * height < kVisionMinimumPixels) {
    SetError(error, "MiniMax H3 keyframe canvas is below the vision minimum");
    return false;
  }
  const auto grid_h = height / 16;
  const auto grid_w = width / 16;
  constexpr int kRowWidth = 3 * 2 * 16 * 16;  // (c, t, y, x) == 1536.
  const std::size_t rows = static_cast<std::size_t>(grid_h) * grid_w;
  output->grid_h = grid_h;
  output->grid_w = grid_w;
  output->pixel_values.assign(rows * kRowWidth, 0.0F);
  for (std::size_t token = 0; token < rows; ++token) {
    // Merge-block-major token order: four consecutive rows form the 2x2
    // block (Qwen2VLImageProcessorFast patchify).
    const int py =
        static_cast<int>(token / 4 / static_cast<std::size_t>(grid_w / 2)) * 2 +
        static_cast<int>(token / 2) % 2;
    const int px =
        static_cast<int>((token / 4) % static_cast<std::size_t>(grid_w / 2)) *
            2 +
        static_cast<int>(token % 2);
    float* row = output->pixel_values.data() + token * kRowWidth;
    for (int c = 0; c < 3; ++c) {
      for (int t = 0; t < 2; ++t) {  // Temporal duplication of one frame.
        for (int y = 0; y < 16; ++y) {
          const std::uint8_t* source =
              rgb.data() +
              (static_cast<std::size_t>(py * 16 + y) * width + px * 16) * 3 + c;
          float* destination =
              row + (static_cast<std::size_t>(c * 2 + t) * 16 + y) * 16;
          for (int x = 0; x < 16; ++x) {
            destination[x] =
                (static_cast<float>(source[static_cast<std::size_t>(x) * 3]) /
                     255.0F -
                 0.5F) /
                0.5F;
          }
        }
      }
    }
  }
  return true;
}

}  // namespace gufo::minimax_h3::image_processor
