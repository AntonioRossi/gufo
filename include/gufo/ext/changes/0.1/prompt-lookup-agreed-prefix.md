# Prompt lookup common continuation

Both `PromptLookup::Find` overloads accept an optional `std::size_t* agreed`
output. For a nonempty match, it gives the length of the continuation that
the competing matches share. A competitor matches at least half as many
tokens as the best match. Rejected source positions also compete.
The default is `nullptr`. Existing calls keep the same result. Extensions
must rebuild. The V4.1 first adapter does not call `Session::DsparkStep`.
