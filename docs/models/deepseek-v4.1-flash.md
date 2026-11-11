# DeepSeek V4.1 Flash 1017

Gufo supports the 384-expert and 192-expert Coder variants through the
DeepSeek V4.1 extension on Linux x86-64 with AMD `gfx1151`.

## Model files and package

Use the six GGUF shards for the selected variant. Give the first shard path
to Gufo. Its name must end with `-00001-of-00006.gguf`. The sixth shard is
the matching `IQ4_NL` Engram file. Keep all six split positions available.
The [implementation notes](../internal/models/deepseek-v4.1-flash/README.md)
identify the selected artifacts and their source revisions.

Use a package that selects the `deepseek_v41_flash` extension. Discovery
mode includes it. The release lock does not include this extension.
See [extension selection](../extensions.md).

## Execution modes

The 384 variant accepts one or two serial HTTP sessions without DSpark.
With DSpark, it accepts one through eight sessions and offers batch widths
2 through 8. The Coder variant accepts one through eight sessions with or
without DSpark. Select the count with `--sessions N`. The default is one
session. Both variants support text, native function tools, and
autoregressive generation.

Use `--stream-experts` to read routed experts through an SSD-backed cache.
Use `--expert-cache-bytes N` to set its byte limit. Zero selects the runtime
default. Dense weights remain resident. The runtime fits the expert cache
after it allocates the selected sessions, so the actual cache can be smaller.

HTTP DSpark execution requires `--speculative dspark` and
`--dspark-model PATH`. The 384 variant requires the default draft controls.
Its `prompt` and `chat` commands support autoregressive execution only.
The Coder variant supports DSpark in HTTP, `prompt`, and `chat`.
The benchmark supports one session, depth zero, and greedy sampling.
See each command's `--help` for its options.

The extension supports RAM prefix reuse. It rejects vision, disk prompt
caching, and the Qwen `--enable-yarn` option. Native tool calls use the V4.1
DSML syntax. The client executes the requested tools.

## Evidence limits

The [check limits](../internal/models/deepseek-v4.1-flash/QUALITY.md) and
[performance limits](../internal/models/deepseek-v4.1-flash/BENCHMARKS.md)
state what the retained evidence establishes. CPU and synthetic kernel
checks do not establish native model behavior or model performance.
