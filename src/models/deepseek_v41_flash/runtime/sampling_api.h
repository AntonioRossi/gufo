#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_SAMPLING_API_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_SAMPLING_API_H_

#include <cstdint>

/* Shared by both expert-count variants; the implementation is compiled once. */
int ds41_sample_argmax(const float* logits, uint32_t vocabulary_size);
int ds41_sample_top_p_min_p(const float* logits, uint32_t vocabulary_size,
                            float temperature, int top_k, float top_p,
                            float min_p, uint64_t* rng_state);

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_SAMPLING_API_H_
