# DeepSeek V4.1 benchmark streaming

`TextBenchOptions` adds `stream_experts` and `expert_cache_bytes`.
The defaults are false and zero. A nonzero cache limit requires streaming.
The benchmark command rejects these options for a provider whose
`supports_expert_streaming` field is false.

The V4.1 benchmark driver forwards both values to `ModelOptions`.
Other providers keep their defaults. Consumers must rebuild against this
header. Prompt, chat, and HTTP sampling defaults do not change.
