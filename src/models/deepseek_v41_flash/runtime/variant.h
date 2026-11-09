#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_VARIANT_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_VARIANT_H_

#include <cstdint>

/* One variant's compile-time shape and identity. The build compiles the
 * variant sources twice with DS41_VARIANT_CODER and
 * DS41_VARIANT_NAMESPACE defined; direct development builds use the 384
 * defaults. */
#ifndef DS41_VARIANT_CODER
#define DS41_VARIANT_CODER 0
#endif
#ifndef DS41_VARIANT_NAMESPACE
#define DS41_VARIANT_NAMESPACE gufo::models::deepseek_v41_flash::v384
#endif

#define DS41_NAMESPACE_BEGIN namespace DS41_VARIANT_NAMESPACE {
#define DS41_NAMESPACE_END }

#if DS41_VARIANT_CODER
DS41_NAMESPACE_BEGIN
inline constexpr uint32_t DS41_VARIANT_EXPERT_COUNT = 192;
inline constexpr char DS41_MODEL_SHAPE_NAME[] = "DeepSeek V4.1 Flash Coder";
inline constexpr char DS41_MODEL_NAME[] =
    "DeepSeek-V4.1-Flash-1017 GSQ-RCO Coder";
inline constexpr char DS41_STATE_ABI_AR[] =
    "deepseek-v41-flash-1017-coder-ar-v1";
inline constexpr char DS41_STATE_ABI_DSPARK[] =
    "deepseek-v41-flash-1017-coder-ds-v1";
/* The local Coder set's backbone shards (1 through 5) and shared Engram. */
inline constexpr uint64_t DS41_TRANSFORMER_BYTES = 72123567456ull;
inline constexpr uint64_t DS41_ENGRAM_BYTES = 110595290816ull;
inline constexpr uint64_t DS41_DSPARK_BYTES = 7976914240ull;
inline constexpr char DS41_DSPARK_SHA256[] =
    "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a";
DS41_NAMESPACE_END
#else
DS41_NAMESPACE_BEGIN
inline constexpr uint32_t DS41_VARIANT_EXPERT_COUNT = 384;
inline constexpr char DS41_MODEL_SHAPE_NAME[] = "DeepSeek V4.1 Flash";
inline constexpr char DS41_MODEL_NAME[] = "DeepSeek-V4.1-Flash-1017";
inline constexpr char DS41_STATE_ABI_AR[] = "deepseek-v41-flash-1017-ar-v1";
inline constexpr char DS41_STATE_ABI_DSPARK[] =
    "deepseek-v41-flash-1017-ds-v1";
inline constexpr uint64_t DS41_TRANSFORMER_BYTES = 0;
inline constexpr uint64_t DS41_ENGRAM_BYTES = 0;
inline constexpr uint64_t DS41_DSPARK_BYTES = 7976914240ull;
inline constexpr char DS41_DSPARK_SHA256[] =
    "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a";
DS41_NAMESPACE_END
#endif

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_VARIANT_H_
