# Acquired-session render-pending decomposition (2026-10-17)

The session converter preserves original source sessions and renders exact
tool-call arguments through the shared serving normalizer. A source argument
that is invalid JSON is an expected, narrowly classified source defect; it is
not permission to guess the intended argument or to abort unrelated valid
sessions.

1. Before rendering a session or writing any graph row, pass its original
messages through the shared `normalize_tool_call_arguments` function. This
performs the same validation as serving and does not alter the retained source.
2. If and only if that function raises `RequestError` with code
   `tool-arguments`, write one durable omission bound to the source file,
   source row, exact normalized row hash, and original assistant target IDs.
   Mark it `source-faithful-review-pending`; do not emit a graph or repair the
   argument.
3. Continue with the next source session. All other errors remain failures and
   stop the incomplete shard; they are not converted into data omissions.
4. Admission independently revalidates the original malformed argument and
   every omission binding. The pending row remains unadmitted and visible in
   accounting until a separately sourced repair exists.

The policy decision is this short source-integrity rule. The crisp helper is
just the shared serving validator, exact receipt hashing, and durable omission
write; it does not choose or infer a replacement argument.
