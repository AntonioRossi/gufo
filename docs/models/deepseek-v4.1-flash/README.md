# DeepSeek V4.1 Flash 1017

This document is a port summary. The
[source model guide](https://github.com/franzmoca/gufo/blob/01fb6f4e39464003efffd97d79607005146280f0/README_DEEPSEEK.md)
describes the source fork at `01fb6f4e39464003efffd97d79607005146280f0`.
Its optional functions exceed this port's scope.

The `deepseek41` provider supports text through `prompt`, `chat`, and the
HTTP text interface. This first port targets Linux x86-64 and AMD `gfx1151`.
Use one session and serial execution. The selected check mode streams
routed experts from SSD with `--stream-experts --expert-cache-bytes 68719476736`.
The source runtime keeps dense weights resident. The cache value is an
upper limit of 64 GiB; the cache can be smaller after session allocation.
The adapter calls `Model::FitExpertCache` after that allocation and reports
the requested limit and actual cache through `event=expert_streaming`.
Without the options, the runtime retains its resident default.
The imported loader reads Engram data from the SSD.

The selected artifact is ISTA GSQ-RCO `IQ1_M` at revision
`65d53bc8be65123e255ff02f3ded958b6dbf54fd`. Give the first shard path to the
engine. The filename must end with `-00001-of-00006.gguf`. All six split
positions must be present. The sixth position is the shared `IQ4_NL`
Engram file under the matching shard-6 name. The bundle contains several
expert types; `IQ1_M` is not the only type.

The adapter rejects vision, tool calls, speculative decoding, and explicit
prompt-cache settings. It does not expose request batching.
Prefix reuse, snapshots, and persistent prompt caching are
unavailable. Each CLI chat turn creates a new session.

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
The probe writes one atomic status file with `--status-json`. It records
the requested streaming values and the actual expert store separately.

See [source attribution and import limits](../../../src/models/deepseek_v41_flash/UPSTREAM.md)
and the [test guide](../../TESTING.md). A discovery build selects this
extension. The first port uses discovery mode. The shipped lock-mode
package, `nix build .#release`, excludes it because `extensions.lock`
does not include it. Select the intended package before model execution.
