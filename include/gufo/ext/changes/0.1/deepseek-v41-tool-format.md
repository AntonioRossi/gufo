# DeepSeek V4.1 tool format

Add `sampling::JsonConstraint::ToolFormat::kDeepSeek41` for the native
DeepSeek V4.1 parameter and call grammar. A text runner can select this
format through `ToolFormat()`. Its extension can register the scanner and
markers through the existing `ToolDialect` contract.

Existing enum values and tool formats retain their behavior.
