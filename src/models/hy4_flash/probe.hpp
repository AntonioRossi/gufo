#pragma once

#include <string>
#include <string_view>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::hy4_flash {

/// True when the opened file is the standard Hy4-Flash Maya-XS IQ2_XXS trunk.
bool ProbeHy4FlashGguf(const core::GgufReader& reader);

/// Opens `model_root` with unsized tensors allowed. Checks metadata only and
/// does not read the pack.
bool ProbeHy4Flash(std::string_view model_root);

/// Checks metadata, requires `model_root` to be the first shard, and verifies
/// the full pack index against the opened shards.
bool VerifyHy4FlashModel(std::string_view model_root, std::string* error);

}  // namespace gufo::models::hy4_flash
