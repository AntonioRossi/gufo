# DeepSeek V4.1 Flash sources

V4.1 semantics (CSA2 KV/index sharing, YoCo layout, shifted mHC, Engram,
sqrtsoftplus routing, the DSpark drafter) follow the official
DeepSeek-V4.1-Flash-1017 `inference/model.py` and `inference/engram.py`, and
the GGUF layout follows the `deepseek41` architecture of the ISTA-DASLab
GSQ-RCO builds.

## Port basis and limits

The source is [franzmoca/gufo at
01fb6f4e39464003efffd97d79607005146280f0](https://github.com/franzmoca/gufo/tree/01fb6f4e39464003efffd97d79607005146280f0/src/models/deepseek_v41_flash).
The port base is Gufo `67dd6354c73fb1325809c7a8f47dd3d30105638c`.
The common ancestor, for attribution, is
`12effe369662ac572cb100cd20157f56a74e2da0`.

The 384 variant supports one or two serial HTTP sessions on Linux `gfx1151`.
The Coder variant supports one through eight sessions. Both support AR and
HTTP DSpark, with resident or streamed experts and RAM prefix reuse.
The 384 prompt and chat paths support AR only. Both variants reject vision
and persistent service prompt caching. This port does not claim conformance.
See the [implementation notes](../../../docs/internal/models/deepseek-v4.1-flash/README.md)
for the selected artifacts and check limits.

Streaming uses `Model::Load`, `Model::FitExpertCache`, and the expert store
from the pinned source. The adapter fits the cache after all selected sessions
and persistent buffers exist. The shared component cache supplies RAM backing.
The source cap, cache-fit calculations, and memory reserves are unchanged.
The added `ExpertCacheStatus` getter reports the store's requested limit,
allocated bytes, and current limit. It does not alter allocation.
The extension manifest declares both options. The host sends explicit values
through `provider_options`. `ResolveExpertOptions` supplies the defaults and
checks the cache condition. Other providers do not accept these options.
The optional probe status and its small stage component are local check
additions. The loader's generic failure reason is `engine_open_failed`.

Native YaRN remains in the imported decoder. The adapter rejects the
`--enable-yarn` option for additional Qwen context scaling.
The runner and both CLI paths use `DeepSeekSamplingDefaults` through the
upstream typed provider interface. It preserves the source sampling values:
temperature 1.0 and top-p 1.0, with explicit overrides preserved.
See `ResolveTextSampling` in the
[source sampling defaults](https://github.com/franzmoca/gufo/blob/01fb6f4e39464003efffd97d79607005146280f0/src/core/text_sampling_defaults.hpp).
The source V4.1 cases in `TestModelSamplingDefaults` are also imported from
[openai_chat_test.cpp](https://github.com/franzmoca/gufo/blob/01fb6f4e39464003efffd97d79607005146280f0/tests/cli/openai_chat_test.cpp).
The CLI context stays at 4096 tokens through `kDefaultContext`.

`bench.cpp` adapts the single-session, depth-zero, greedy path of
`RunDeepSeekBenchmark` from the pinned source
[benchmark command](https://github.com/franzmoca/gufo/blob/01fb6f4e39464003efffd97d79607005146280f0/src/cli/bench/bench.cpp).
It preserves the source token pattern, timing boundaries, repetition order,
and mean with sample standard deviation. Decode starts from 16 tokens at
depth zero. The model and expert store remain alive across repetitions.
The benchmark does not call `FitExpertCache`, as in the source benchmark.
It reports the existing `ExpertCacheStatus` outside the measured intervals.
`TextBenchOptions::provider_options` forwards the streaming flag and cache cap. Sampled
decoding, nonzero depth, concurrency, and prefill validation are not exposed
by this first benchmark path. Prompt, chat, and HTTP paths are unchanged.

The host target contains the chat template, tool scanner, sampling, probe,
and registration. `ConvertChat` maps text requests and tool history into the
source template types. The runner selects the registered `deepseek41` dialect.
`DeepSeekToolGrammar` supplies its syntax to the upstream grammar compiler.
The HTTP parser uses the registered dialect and the existing response and
streaming paths. Required and forced tool selection use the shared server
constraints. Image input remains unsupported.
The HIP target contains the engine, runtime, adapter, and non-MMQ kernels.
The MMQ target keeps its source compiler settings and generated type objects.
The import uses the public forwarding headers for core declarations.
The loader, expert store, and ROCm backend use the current core log header
through `gufo/ext/host.hpp`.

The import excludes `vision/` and the source continuation adapter.
Both variants use the upstream window-row component contract. The adapter
uses V4.1 native state parts and separate FP4 code and scale components.
Required pooling state remains private. Service disk caching stays disabled.
The import also excludes the unused `reference/`,
`artifact_hint.hpp`, and `dspark_sampler.hpp`.
The source scanner uses the existing extension argument and call types.
Its parsing rules are unchanged. The source CMake
file is replaced by `extension.cmake`.
The selected test source keeps its cases and assertions. Formatting follows
the port base. `fixtures/encoding-1017.json` is copied without changes.
Test target links use the new HIP target and `hip::host`. The MMQ test also
links the MMQ target for its header path.

The independent CPU oracle and experiment records remain in the source:
[tools/ds41](https://github.com/franzmoca/gufo/tree/01fb6f4e39464003efffd97d79607005146280f0/tools/ds41)
and [EXPERIMENTS.md](https://github.com/franzmoca/gufo/blob/01fb6f4e39464003efffd97d79607005146280f0/docs/models/deepseek-v4.1-flash/EXPERIMENTS.md).

## Origin

The package started as a copy of Gufo's `src/models/deepseek_v4_flash`
engine, which is adapted from DS4 commit
`84cc882352757baf628a1776badf7cc54d584e28` ([LICENSE.ds4](LICENSE.ds4)). The
V4 graph, session and kernel layer have since been replaced and removed; the
GGUF loader (`runtime/model_data.cpp`), the tokenizer (`runtime/tokenizer.cpp`)
and the resident weight upload (`kernels/rocm/backend.hip.cpp`) still derive
from it. The V4 engine itself stays unchanged.

## Layout

- `engine.hpp` and `engine.cpp` are the Gufo-owned C++20 API that product code
  uses for model loading, tokenization, request sessions, snapshots, logits
  and cancellation.
- `runtime/` holds the V4.1 decoder (`v41_decode.cpp`), request sessions and
  the DSpark policy (`v41_session.cpp`), the drafter (`v41_dspark.cpp`), SSD
  expert streaming (`expert_store.cpp`), GGUF loading and the tokenizer.
- `kernels/rocm/` holds every V4.1 kernel (`ds41_v41.h` lists them) and the
  vendored llama.cpp quantized matrix kernels in `mmq/`
  ([VENDOR.md](kernels/rocm/mmq/VENDOR.md)).
- `runner.cpp` connects ordinary generation to the current text interface.
- `drivers.cpp` connects the prompt and chat commands to that runner. Each
  chat turn creates a new session.
- `bench.cpp` connects the selected source benchmark path to `gufo bench`.

DeepSeek kernels are private to this package: a change here must not alter
another model's output or performance.
