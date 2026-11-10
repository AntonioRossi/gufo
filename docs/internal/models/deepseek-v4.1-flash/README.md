# DeepSeek V4.1 Flash 1017

This document is a port summary. The
[source model guide](https://github.com/franzmoca/gufo/blob/01fb6f4e39464003efffd97d79607005146280f0/README_DEEPSEEK.md)
describes the source fork at `01fb6f4e39464003efffd97d79607005146280f0`.
Its optional functions exceed this port's scope.

The extension has two `deepseek41` variants in one executable. The 384-expert
IQ1_M variant uses one or two serial sessions. The 192-expert Coder variant
uses one through eight sessions in AR or DSpark. Both variants target Linux x86-64 and
AMD `gfx1151`, and both use the `prompt`, `chat`, and HTTP text interfaces.
Select sessions with `--sessions N` or `-j N`, within the variant limit.
The default remains one.

The selected 384 artifact is ISTA GSQ-RCO `IQ1_M` at revision
`65d53bc8be65123e255ff02f3ded958b6dbf54fd`. The Coder artifact is ISTA GSQ-RCO
`IQ2_XXS` at local revision `54229d38d588a66d1c6cff08af7b281b9326c656`.
Give the first shard path to the engine. The filename must end with
`-00001-of-00006.gguf`. All six split positions must be present. The sixth
position is the shared `IQ4_NL` Engram file under the matching shard-6 name.
The bundles contain several routed expert types; neither is uniform.

The selected check mode streams routed experts from SSD with
`--stream-experts --expert-cache-bytes 68719476736` for the 384 variant.
The source runtime keeps dense weights resident. The cache value is an
upper limit; the cache can be smaller after session allocation. The adapter
allocates every selected session and its long-lived buffers before
`Model::FitExpertCache`, then reports the requested limit and actual cache
through `event=expert_streaming`. Without the options, the runtime retains
its resident default. The imported loader reads Engram data from the SSD.

The HTTP interface supports function tools, assistant tool-call history,
and tool results. It uses the source V4.1 DSML grammar and scanner.
Tool choice can be `auto`, `none`, `required`, or a declared function.
The shared server retains namespaces and the parallel-call control.
Responses preserve assistant text and reasoning. Calls with no assistant
text use `content: null`. Completed calls use `finish_reason: "tool_calls"`;
length limits and stop sequences keep their own finish reasons.
Streaming responses send incremental tool arguments through the existing
tool-delta path. Schema enforcement has the same strict and non-strict
limits as the source grammar. The client executes the tools and returns
their results; the model server does not execute them.

The adapter rejects vision and disk prompt caching. The 384 HTTP adapter
accepts DSpark with the pinned Q8_0 drafter, one or two serial sessions,
and default draft controls. It rejects other speculative backends and request batching.
It accepts a positive RAM prompt-cache capacity. Its DSpark state ABI is
`deepseek-v41-flash-1017-ds-v1`. The 384 CLI prompt and chat paths still
reject speculative decoding. The Coder
accepts `--speculative dspark` with `--dspark-model`; its sessions are
speculative and its AR requests can use batch widths 2 through 8. Each CLI
chat turn resets the pooled session before the new prompt.

Both variants' HTTP requests use the shared component cache. Each decoder
layer and each DSpark stage has a window component for its trailing 128 rows.
The native rings have no extra capacity, so each chunk contains one row.
Transfers use absolute row numbers and split at the ring boundary.
The unfinished compressor groups remain private. Ratio-one sources need no
pooling component. FP4 attention codes, indexer codes, and their scales keep
separate append components. Tokens, logits, range metadata, and request state
complete a checkpoint. Layout version 2 separates these records from the
earlier private-window layout. A restore must supply every component and the
retained window rows at the same valid boundary before execution.
Decoder coverage and the DSpark range retain their separate checks.
The native restore
reconstructs decoder position and Engram history from evaluated tokens.
The runner resets request lookup and acceptance policy before a new request;
session timing estimates keep their native behavior.

`--cache-ram-bytes 0` selects automatic sizing. A positive value selects a
retained-cache cap. `off` disables retained RAM checkpoints and permits reuse
of live state. `cache_prompt: false` bypasses prefix lookup for an uncached
comparison. It can still populate the cache. A growing prompt reuses a usable
checkpoint boundary and prefills the remaining suffix. This cache does not
change expert storage. The 384 adapter supports retained checkpoints and
restored-prefix execution with one or two serial sessions. A configured
context or cache cap does not establish tested depth or host capacity.

The imported decoder keeps native YaRN. The `--enable-yarn` option is
unsupported; that option selects additional context scaling for Qwen.
The CLI prompt and chat paths use `kDefaultContext = 4096` tokens.
The HTTP path uses the model's native context when `--context` is unset.
An explicit HTTP `--context` selects its capacity separately.

The runner and both CLI paths use `TextModelPreset::kDeepSeekV41Flash`.
Its defaults are `temperature=1.0` and `top_p=1.0`, with thinking on or off.
Explicit sampling values override the preset. See `ResolveTextSampling`
in the [source sampling defaults](https://github.com/franzmoca/gufo/blob/01fb6f4e39464003efffd97d79607005146280f0/src/core/text_sampling_defaults.hpp).
These code defaults do not approve a launch configuration.

`gufo bench` supports one session, depth zero, and greedy ordinary decoding.
It forwards `--stream-experts` and `--expert-cache-bytes` to the runtime.
The benchmark keeps its greedy defaults; it does not use the serving preset
or a chat template. At depth zero, `tg128` starts after a 16-token prefix.
The `pp2048` and `tg128` tests use separate sessions in one process.
For these workloads, context is 4096 and the default prefill batch is 2048.
Every repetition is measured. Output gives the mean and sample standard
deviation. This path rejects nonzero depth, concurrent sessions, sampled or
speculative decoding, `--validate-prefill`, and `--enable-yarn`.

This port does not claim conformance. The selected checks without model
files are `ds41.template`, `qwen38_flash_next.prompt_lookup`,
`ds41.mmq_typed`, `ds41.attention`, `ds41.hc_state_pass`, and
`ds41.candidate_mask`. The last four need a `gfx1151` HIP device.
These checks do not establish model loading or HTTP generation.
The CPU `ds41.probe_status` check covers the optional probe status component.
The CPU `ds41.continuation_layout` check covers Coder component positions,
FP4 code and scale sizes, private state sizes, and backing allocation.
The HIP `ds41.continuation_adapter` check needs
`GUFO_DEEPSEEK_V41_CODER_MODEL` and `GUFO_DEEPSEEK_V41_CODER_DSPARK_MODEL`.
It loads the resident Coder once at context 8192. It checks AR and DSpark
restores, frontier logits, continued tokens, and rejection of partial state.
It uses a full-session snapshot only as a reference for the restore check.
The CPU `ds41.tool_calls` check covers request conversion, the scanner,
and required-tool grammar. The shared HTTP and constraint tests include
the source V4.1 cases, including incremental and interrupted calls.
The probe writes one atomic status file with `--status-json`. It records
the requested streaming values and the actual expert store separately.

The provider registers the V4.1 tool syntax and scanner through
`RegisterToolDialect`. Expert controls use `provider_options`. Sampling and
DSpark defaults use the upstream provider interfaces. `Runner::Prefill`
uses the full prompt end for bounded replay. `Runner::CanContinue` lets the
shared cache reject a restored frontier with insufficient decoder coverage.
Layer yields stay disabled because sessions share live activation buffers.

See [source attribution and import limits](../../../../src/models/deepseek_v41_flash/UPSTREAM.md)
and the [test guide](../../testing.md). A discovery build selects this
extension. The first port uses discovery mode. The shipped lock-mode
package, `nix build .#release`, excludes it because `extensions.lock`
does not include it. Select the intended package before model execution.
