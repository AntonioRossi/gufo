#include "provider.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <gufo/ext/generated/hy4_flash_constants.hpp>
#include <gufo/ext/image.hpp>
#include <gufo/ext/options.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <hy4_kernels.cuh>
#include <hy4_model.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "image_prep.hpp"
#include "logit_check.hpp"
#include "tokenizer.hpp"

namespace gufo::models::hy4_flash {
namespace {

int Argmax(const float* logits, int size) {
  int best = 0;
  for (int i = 1; i < size; ++i) {
    if (logits[i] > logits[best]) {
      best = i;
    }
  }
  return best;
}

bool LoadHy4Model(const std::filesystem::path& pack, const std::string& draft,
                  bool verbose, hy4::Model* model, std::string* error) {
  hy4::Options options;
  options.pack = pack.string();
  options.gpus = {0};
  options.main_gpu = 0;
  options.max_ctx = 524288;
  options.prefill_chunk = 1024;
  options.tp = 0;
  options.vision = true;
  options.mtp = false;
  options.dfly2 = draft == "dfly2";
  options.verbose = verbose;
  if (!model->load(options, *error)) {
    return false;
  }
  if (draft == "dfly2" && !model->has_dfly2()) {
    *error = "the pack has no DFly2 drafter";
    return false;
  }
  return true;
}

}  // namespace

static int RunHy4Prompt(const ext::TextPromptOptions& options,
                        const core::GgufReader& reader,
                        std::chrono::steady_clock::time_point load_start) {
  (void)load_start;

  std::string draft = "none";
  if (const auto* value = ext::FindOption<std::string>(options.provider_options,
                                                       "--hy4-draft")) {
    draft = *value;
  }
  if (draft != "none" && draft != "dfly2") {
    std::cerr << "Error: --hy4-draft must be none or dfly2\n";
    return 2;
  }
  if (options.use_chat_template) {
    std::cerr << "hy4: chat templates are not supported\n";
    return 2;
  }
  if (!options.system_prompt.empty()) {
    std::cerr << "hy4: system prompts are not supported\n";
    return 2;
  }
  if (options.reasoning_mode != "auto" || options.reasoning_effort != "auto" ||
      options.preserve_thinking != "auto") {
    std::cerr << "hy4: reasoning controls are not supported\n";
    return 2;
  }
  if (!options.sampling.can_use_unmodified_argmax()) {
    std::cerr << "hy4: only greedy sampling is supported\n";
    return 2;
  }
  if (!options.mtp_model_path.empty()) {
    std::cerr << "hy4: --mtp-model is not supported; use --hy4-draft dfly2\n";
    return 2;
  }
  if (!options.dflash_model_path.empty()) {
    std::cerr << "hy4: --dflash-model is not supported; "
                 "use --hy4-draft dfly2\n";
    return 2;
  }
  if (!options.dspark_model_path.empty()) {
    std::cerr << "hy4: --dspark-model is not supported; "
                 "use --hy4-draft dfly2\n";
    return 2;
  }
  if (!options.speculative_backend.empty()) {
    std::cerr << "hy4: --speculative is not supported; "
                 "use --hy4-draft dfly2\n";
    return 2;
  }
  if (options.force_cpu) {
    std::cerr << "hy4: the CPU execution mode is not supported\n";
    return 2;
  }
  if (!options.vision_model_path.empty()) {
    std::cerr << "hy4: a separate vision model is not supported\n";
    return 2;
  }
  if (options.add_vision_id) {
    std::cerr << "hy4: the vision ID option is not supported\n";
    return 2;
  }

  const std::filesystem::path first_shard(options.model_path);
  std::string error;
  if (!VerifyHy4FlashPack(options.model_path, reader, &error)) {
    std::cerr << "hy4: pack verification failed: " << error << "\n";
    return 1;
  }
  if (options.image_paths.size() > 1) {
    std::cerr << "hy4: one image is supported\n";
    return 2;
  }
  const std::filesystem::path pack = first_shard.parent_path() / "maya-hy4";

#ifdef HY4_TEST_DIAGNOSTICS
  const char* kld_reference = std::getenv("HY4_TEST_KLD_REFERENCE");
  const char* kld_out = std::getenv("HY4_TEST_KLD_OUT");
  if (kld_reference != nullptr && *kld_reference != '\0') {
    TopkReference reference;
    if (!LoadTopkReference(kld_reference, 64, &reference, &error)) {
      std::cerr << "hy4: " << error << "\n";
      return 1;
    }
    hy4::Model model;
    if (!LoadHy4Model(pack, "none", options.verbose, &model, &error)) {
      std::cerr << "hy4: load: " << error << "\n";
      return 1;
    }
    if (reference.n_ctx > model.max_ctx()) {
      std::cerr << "hy4: the reference context exceeds the model context\n";
      return 1;
    }
    const int vocab = model.vocab();
    const int first = reference.n_ctx / 2;
    std::vector<float> logits(static_cast<std::size_t>(hy4::k::kMaxWin) *
                              static_cast<std::size_t>(vocab));
    const bool write_output = kld_out != nullptr && *kld_out != '\0';
    std::ofstream out;
    if (write_output) {
      out.open(kld_out);
      if (!out) {
        std::cerr << "hy4: cannot open the KLD output\n";
        return 1;
      }
      out << std::setprecision(9);
    }
    std::vector<double> kls;
    long positions = 0;
    long same = 0;
    double kl_sum = 0;
    for (int r = 0; r < reference.rows; ++r) {
      const int* ids = reference.tokens.data() +
                       static_cast<std::size_t>(r) * reference.n_ctx;
      if (!model.forward(ids, first, 0, 0, nullptr, error)) {
        std::cerr << "hy4: kld forward: " << error << "\n";
        return 1;
      }
      for (int p = first; p < reference.n_ctx - 1; p += hy4::k::kMaxWin) {
        const int T = std::min(hy4::k::kMaxWin, reference.n_ctx - 1 - p);
        if (!model.forward(ids + p, T, p, T, logits.data(), error)) {
          std::cerr << "hy4: kld forward at " << p << ": " << error << "\n";
          return 1;
        }
        for (int t = 0; t < T; ++t) {
          const int pos = p + t;
          if (reference
                  .mask[static_cast<std::size_t>(r) * (reference.n_ctx - 1) +
                        pos] == 0) {
            continue;
          }
          const std::size_t index =
              (static_cast<std::size_t>(r) * reference.n_ctx + pos) *
              reference.k;
          KldPosition position;
          if (!ScorePosition(
                  logits.data() + static_cast<std::size_t>(t) * vocab, vocab,
                  reference.reference_log_probs.data() + index,
                  reference.ids.data() + index, reference.k, &position,
                  &error)) {
            std::cerr << "hy4: kld score: " << error << "\n";
            return 1;
          }
          if (write_output) {
            WriteKldPosition(out, r, pos, position);
            if (!out) {
              std::cerr << "hy4: KLD output write failed\n";
              return 1;
            }
          }
          kl_sum += position.kl;
          same += position.top1 ? 1 : 0;
          ++positions;
          kls.push_back(position.kl);
        }
      }
    }
    if (positions == 0) {
      std::cerr << "hy4: the reference has no masked scoring positions\n";
      return 1;
    }
    if (write_output) {
      out.close();
      if (!out) {
        std::cerr << "hy4: KLD output close failed\n";
        return 1;
      }
    }
    std::sort(kls.begin(), kls.end());
    std::cerr << "hy4 kld_topk: " << positions << " positions, mean "
              << kl_sum / positions << ", median " << kls[kls.size() / 2]
              << ", p99 " << kls[static_cast<std::size_t>(kls.size() * 0.99)]
              << ", top-1 " << 100.0 * same / positions << "%\n";
    return 0;
  }
#endif

  if (options.prompt_text.empty()) {
    std::cerr << "hy4: the prompt is empty\n";
    return 1;
  }

  auto tokenizer = Tokenizer::Load(pack / "tokenizer", &error);
  if (!tokenizer) {
    std::cerr << "hy4: tokenizer: " << error << "\n";
    return 1;
  }
  const std::vector<int> input = tokenizer->Encode(options.prompt_text);
  if (input.empty()) {
    std::cerr << "hy4: the prompt encoded to no tokens\n";
    return 1;
  }
  if (options.display_prompt) {
    std::cout << options.prompt_text;
    if (options.prompt_text.empty() || options.prompt_text.back() != '\n') {
      std::cout << '\n';
    }
  }

  hy4::Model model;
  if (!LoadHy4Model(pack, draft, options.verbose, &model, &error)) {
    std::cerr << "hy4: load: " << error << "\n";
    return 1;
  }
  if (options.verbose) {
    std::cerr << "hy4 max_ctx: " << model.max_ctx() << "\n";
    std::cerr << "hy4 chunk: " << model.chunk() << "\n";
  }

  std::vector<int> ids = input;
  if (!options.image_paths.empty()) {
    std::vector<std::uint8_t> encoded;
    try {
      encoded = gufo::core::ReadImageFile(options.image_paths[0]);
    } catch (const std::exception& exception) {
      std::cerr << "hy4: image: " << exception.what() << "\n";
      return 1;
    }
    const auto* geometry = model.vision();
    if (geometry == nullptr) {
      std::cerr << "hy4: the pack has no vision tower\n";
      return 1;
    }
    const VisionGeometry prep_geometry{geometry->patch, geometry->merge,
                                       geometry->min_pixels,
                                       geometry->max_pixels};
    gufo::core::Image picture;
    if (!PrepareImage(encoded, prep_geometry, &picture, &error)) {
      std::cerr << "hy4: image: " << error << "\n";
      return 1;
    }
    const int tokens = geometry->tokens(static_cast<int>(picture.width),
                                        static_cast<int>(picture.height));
    if (tokens <= 0) {
      std::cerr << "hy4: unsupported image dimensions\n";
      return 1;
    }
    std::vector<float> rows;
    if (!model.encode_image(picture.pixels.data(),
                            static_cast<int>(picture.width),
                            static_cast<int>(picture.height), rows, error)) {
      std::cerr << "hy4: image encode: " << error << "\n";
      return 1;
    }
    if (rows.size() != static_cast<std::size_t>(tokens) *
                           static_cast<std::size_t>(geometry->out_dim)) {
      std::cerr << "hy4: image rows do not match the geometry\n";
      return 1;
    }
#ifdef HY4_TEST_DIAGNOSTICS
    if (const char* dump = std::getenv("HY4_TEST_IMAGE_DUMP");
        dump != nullptr && *dump != '\0') {
      const std::filesystem::path directory(dump);
      std::error_code create_error;
      std::filesystem::create_directories(directory, create_error);
      if (create_error) {
        std::cerr << "hy4: image dump directory: " << create_error.message()
                  << "\n";
        return 1;
      }
      std::ofstream rgb(directory / "rgb.bin", std::ios::binary);
      rgb.write(reinterpret_cast<const char*>(picture.pixels.data()),
                static_cast<std::streamsize>(picture.pixels.size()));
      rgb.close();
      if (!rgb) {
        std::cerr << "hy4: image dump write failed\n";
        return 1;
      }
      std::ofstream size(directory / "size.txt");
      size << picture.width << "x" << picture.height;
      size.close();
      if (!size) {
        std::cerr << "hy4: image dump size write failed\n";
        return 1;
      }
      std::ofstream rows_file(directory / "rows.f32", std::ios::binary);
      rows_file.write(
          reinterpret_cast<const char*>(rows.data()),
          static_cast<std::streamsize>(rows.size() * sizeof(float)));
      rows_file.close();
      if (!rows_file) {
        std::cerr << "hy4: image dump rows write failed\n";
        return 1;
      }
    }
#endif
    hy4::Model::Image image;
    ids.clear();
    ids.push_back(geometry->image_start);
    image.start = static_cast<int64_t>(ids.size());
    ids.insert(ids.end(), static_cast<std::size_t>(tokens),
               geometry->image_token);
    ids.push_back(geometry->image_end);
    ids.insert(ids.end(), input.begin(), input.end());
    image.rows = std::move(rows);
    model.set_images({std::move(image)});
  }

  const int64_t max_ctx = model.max_ctx();
  if (ids.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      options.max_tokens >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    std::cerr << "hy4: token count is too large\n";
    return 2;
  }
  if (static_cast<int64_t>(ids.size()) +
          static_cast<int64_t>(options.max_tokens) >
      max_ctx) {
    std::cerr << "hy4: the prompt and output limit exceed the context\n";
    return 2;
  }
  if (options.verbose) {
    std::cerr << "hy4 input_ids:";
    for (const int id : ids) {
      std::cerr << " " << id;
    }
    std::cerr << "\n";
  }

  const int vocab = model.vocab();
  std::vector<float> logits(static_cast<std::size_t>(vocab));
  const auto prefill_start = std::chrono::steady_clock::now();
  if (!model.forward(ids.data(), static_cast<int>(ids.size()), 0, 1,
                     logits.data(), error)) {
    std::cerr << "hy4: prefill: " << error << "\n";
    return 1;
  }
  const auto prefill_end = std::chrono::steady_clock::now();

  std::vector<int> output;
  const int limit = static_cast<int>(options.max_tokens);
  int64_t committed = static_cast<int64_t>(ids.size());
  const auto decode_start = prefill_end;
  if (draft == "dfly2") {
    const int n_draft = std::min(model.draft_default(), model.draft_max());
    if (n_draft <= 0) {
      std::cerr << "hy4: the drafter has no drafts\n";
      return 1;
    }
    const float floor = 0.8F;
    std::vector<float> lw(static_cast<std::size_t>(n_draft + 1) *
                          static_cast<std::size_t>(vocab));
    std::vector<int> drafts;
    std::vector<float> confs;
    std::vector<int> win;
    const auto top_of = [&](const float* row, int r) {
      const int top = model.row_top(r);
      return top >= 0 ? top : Argmax(row, vocab);
    };
    int64_t pos = static_cast<int64_t>(ids.size());
    int tok = top_of(logits.data(), 0);
    int rounds = 0;
    while (static_cast<int>(output.size()) < limit) {
      output.push_back(tok);
      if (tok == model.eos() || static_cast<int>(output.size()) >= limit) {
        break;
      }
      const int remaining = limit - static_cast<int>(output.size());
      const int n = std::min(n_draft, remaining);
      if (!model.draft(tok, pos, n, floor, drafts, confs, error)) {
        std::cerr << "hy4: draft: " << error << "\n";
        return 1;
      }
      int k = 0;
      while (k < static_cast<int>(drafts.size()) &&
             confs[static_cast<std::size_t>(k)] >= floor) {
        ++k;
      }
      win.assign(1, tok);
      win.insert(win.end(), drafts.begin(), drafts.begin() + k);
      if (!model.forward(win.data(), k + 1, pos, k + 1, lw.data(), error)) {
        std::cerr << "hy4: verify: " << error << "\n";
        return 1;
      }
      ++rounds;
      ++pos;
      int accepted_round = 0;
      int i = 0;
      for (; i < k; ++i) {
        if (top_of(lw.data() + static_cast<std::size_t>(i) * vocab, i) !=
            drafts[static_cast<std::size_t>(i)]) {
          break;
        }
        output.push_back(drafts[static_cast<std::size_t>(i)]);
        ++accepted_round;
        if (drafts[static_cast<std::size_t>(i)] == model.eos()) {
          break;
        }
        ++pos;
        if (static_cast<int>(output.size()) >= limit) {
          break;
        }
      }
      if (options.verbose) {
        std::cerr << "hy4 dfly2 round " << rounds << ": drafts";
        for (int d = 0; d < k; ++d) {
          std::cerr << " " << drafts[static_cast<std::size_t>(d)];
        }
        std::cerr << "; accepted " << accepted_round << "\n";
      }
      if (output.back() == model.eos() ||
          static_cast<int>(output.size()) >= limit) {
        break;
      }
      tok = top_of(lw.data() + static_cast<std::size_t>(i) * vocab, i);
    }
    committed = pos;
  } else {
    for (int i = 0; i < limit; ++i) {
      const int next = Argmax(logits.data(), vocab);
      output.push_back(next);
      if (next == model.eos()) {
        break;
      }
      if (!model.forward(&next, 1, committed, 1, logits.data(), error)) {
        std::cerr << "hy4: decode: " << error << "\n";
        return 1;
      }
      ++committed;
    }
  }
  const auto decode_end = std::chrono::steady_clock::now();

  std::vector<int> displayed = output;
  if (!displayed.empty() && displayed.back() == model.eos()) {
    displayed.pop_back();
  }
  std::cout << tokenizer->Decode(displayed);

#ifdef HY4_TEST_DIAGNOSTICS
  const bool committed_diagnostic =
      std::getenv("HY4_TEST_COMMITTED") != nullptr;
#else
  const bool committed_diagnostic = false;
#endif
  if (options.verbose || committed_diagnostic) {
    std::cerr << "hy4 output_ids:";
    for (const int id : output) {
      std::cerr << " " << id;
    }
    std::cerr << "\n";
    const char* stop = output.empty()
                           ? "empty"
                           : (output.back() == model.eos() ? "eos" : "limit");
    std::cerr << "hy4 stop: " << stop << "\n";
    const double prefill_s =
        std::chrono::duration<double>(prefill_end - prefill_start).count();
    const double decode_s =
        std::chrono::duration<double>(decode_end - decode_start).count();
    std::cerr << "hy4 prefill_s: " << prefill_s << "\n";
    std::cerr << "hy4 decode_s: " << decode_s << "\n";
  }
  if (committed_diagnostic) {
    std::cerr << "committed_token_count " << committed << "\n";
  }

  model.answered(static_cast<int64_t>(ids.size() + output.size()) - 1);
  std::string profile_error;
  if (!model.save_profile(profile_error)) {
    std::cerr << "hy4: expert profile: " << profile_error << "\n";
  }
  return 0;
}

}  // namespace gufo::models::hy4_flash

void hy4_flash_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_hy4_flash::kProvider_hy4_flash);
  auto provider = gufo::models::hy4_flash::Hy4FlashTextProvider();
  provider.prompt_driver = &gufo::models::hy4_flash::RunHy4Prompt;
  registry.AddTextProvider(std::move(provider));
}