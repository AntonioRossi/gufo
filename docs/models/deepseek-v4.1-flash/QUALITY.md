# DeepSeek V4.1 Flash check limits

The selected kernel tests use synthetic inputs and the source assertions.
The chat-template test uses the source `encoding-1017.json` fixture.
These tests do not establish full-model correctness, output quality, or
HTTP behavior. This port does not claim conformance.

See the [selected tests and scope](README.md). The base prompt-lookup test
does not check the new `agreed` output. The first adapter does not call
`Session::DsparkStep` and does not use that output.
