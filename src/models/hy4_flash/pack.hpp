#pragma once

#include <filesystem>
#include <string>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::hy4_flash {

/// Compares every entry of the pack index with the opened GGUF tensors: name,
/// type, dimensions, payload, shard and data address. Rejects unknown,
/// duplicate and omitted tensors.
bool VerifyPack(const std::filesystem::path& pack_dir,
                const core::GgufReader& reader, std::string* error);

}  // namespace gufo::models::hy4_flash
