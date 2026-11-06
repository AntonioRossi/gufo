/* Print the two V4.1 probes' verdicts for one GGUF file. This is a development
 * check, not a CTest. */
#include <cstdio>
#include <string>

#include "src/models/deepseek_v41_flash/probe.hpp"

int main(int argc, char** argv) {
  if (argc != 2)
    return 2;
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 3;
  }
  std::printf(
      "v384=%d coder=%d\n",
      gufo::models::deepseek_v41_flash::ProbeDeepSeekV384Gguf(*reader),
      gufo::models::deepseek_v41_flash::ProbeDeepSeekCoderGguf(*reader));
  return 0;
}