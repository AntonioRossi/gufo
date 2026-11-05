# DeepSeek V4.1 integration

Keep the fork extension fields in API series 0.2:

- `TextPromptOptions`, `TextRunnerOptions`, and `TextBenchOptions` keep
  `stream_experts` and `expert_cache_bytes`.
- `TextProvider` keeps `supports_expert_streaming`.
- `TextModelPreset` keeps `kDeepSeekV41Flash`. Its temperature and top-p
  defaults remain 1.0 for each thinking mode.
- `JsonConstraint::ToolFormat` keeps `kDeepSeek41` for the native tool grammar.
- `PromptLookup::Find` keeps its optional `agreed` output.

V4.1 uses the component slot interface without prompt-cache components.
A runner with `prefix_reuse = false` does not register a prefix inventory
or create checkpoint history. Capture and restore remain unsupported.
