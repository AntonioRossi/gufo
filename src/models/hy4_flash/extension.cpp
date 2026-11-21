#include <gufo/ext/registry.hpp>

#include "provider.hpp"

void hy4_flash_register(gufo::ext::Registry& registry) {
  registry.AddTextProvider(gufo::models::hy4_flash::Hy4FlashTextProvider());
}
