#pragma once

#include <gufo/ext/text.hpp>

#include "probe.hpp"

namespace gufo::models::hy4_flash {

/// The text provider shared by the Stage 0 CPU entry and the Stage 1 HIP
/// entry. The HIP entry adds its drivers to a copy of this value.
inline ext::TextProvider Hy4FlashTextProvider() {
  return {
      .id = "hy4-flash",
      .probe = &ProbeHy4Flash,
      .probe_gguf = &ProbeHy4FlashGguf,
      .accepts_unsized_tensors = true,
  };
}

}  // namespace gufo::models::hy4_flash
