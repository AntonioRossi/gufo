# DeepSeek V4.1 sampling preset

`TextModelPreset` adds `kDeepSeekV41Flash`. `TextPreset` selects it for
the `deepseek41` architecture. `ResolveTextSampling` uses temperature 1.0
and top-p 1.0 with thinking on or off. Explicit sampling overrides and
output constraints remain in effect. Existing presets keep their behavior.
Extensions that use the new enum value must rebuild with this API revision.
