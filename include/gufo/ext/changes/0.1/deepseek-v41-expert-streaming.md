# DeepSeek V4.1 expert streaming

`TextRunnerOptions` and `TextPromptOptions` add `stream_experts` and
`expert_cache_bytes`. The defaults are false and zero. A nonzero cache limit
requires streaming. `TextProvider::supports_expert_streaming` defaults to
false. The prompt, chat, and HTTP paths reject streaming options when the
selected provider does not support them.

The V4.1 provider forwards both values to its existing runtime. Other
providers keep their defaults. Consumers must rebuild against this header.
