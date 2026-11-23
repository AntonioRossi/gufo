# Image alpha compositing and shared RGB8 resampler

Image (`src/core/image.hpp`):
- `ImageDecodeOptions` selects the white alpha composite for PNG colour types
  3, 4, and 6 and for WebP, and the WebP EXIF orientation.
- `DecodeImage(bytes, options)` applies those choices. The existing
  `DecodeImage(bytes)` entry point keeps the current behavior.
- `ResizeRgb8` publishes the Pillow-parity two-pass Lanczos RGB8 resize.

The additions are additive. Existing callers keep compiling. Runners that
decode images can select the new options. Extensions can use the shared
resampler. Runners and extensions must be rebuilt against the new header.