# DeepSeek V4.1 provider interfaces

The DeepSeek V4.1 extension uses the upstream provider interfaces.
This change replaces the fork fields from `deepseek-v41-integration.md`
and the earlier DeepSeek sampling preset. Their records remain historical.

- Remove `stream_experts` and `expert_cache_bytes` from `TextRunnerOptions`,
  `TextPromptOptions`, and `TextBenchOptions`. Remove
  `TextProvider::supports_expert_streaming`. Declare the two command options
  in the extension manifest. Read their explicit values from
  `provider_options`. The provider keeps the default values and checks that
  a nonzero cache limit has expert streaming enabled.
- Remove the fork `ToolFormat::kDeepSeek41` value. Register `deepseek41`
  with its `ToolSyntax`, markers, and scanner. Select that identity through
  `TextModelRunner::ToolDialectId()`.
- Remove `TextModelPreset::kDeepSeekV41Flash`. Supply typed sampling defaults
  from the provider and runner. Temperature and top-p remain 1.0. Explicit
  caller values keep their precedence.

These source changes require an update to callers of the removed fork
fields and enum values. The command option names and native tool syntax
stay unchanged. Integer provider options use signed 64-bit values. The
expert-cache option rejects negative values and accepts the approved limits.
