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

## Opt-in source-syntax repair view (2026-10-18)

Some pending D32 rows contain a narrowly recognizable serialization defect:
the JSON object and its `query` value boundaries are intact, but the literal
query text was emitted without JSON string quoting. For those rows only, a
derived sibling view may replace the malformed argument string with a valid
JSON encoding of the exact bounded query slice. The original normalized source
and raw parquet remain immutable and authoritative.

1. A repair candidate must match a declared schema and unique delimiters, with
   intact required fields and no ambiguous/truncated syntax. The repair may
   only add the JSON quoting/escaping required to encode the exact original
   query characters. It must not execute the text, normalize it, infer missing
   content, or alter any other field.
2. The derived view records the original source file/row/session and argument
   bytes and hashes, the exact repaired argument and hash, repair rule/version,
   and the transformation classification `source-syntax-repair-best-effort`.
   This does not claim that the repaired argument equals the producer's intent
   or that the tool call was executed successfully.
3. Both model renderers consume the same source-bound repair view and retain
   the complete trajectory, split, and source groups. Standard current global
   held/quality protection and admission run on the derived view; graph IDs
   must distinguish the repaired rendered view from any legacy view.
4. Re-render/admit only rows whose original argument hashes and source
   identities match the repair manifest. Unmatched rows, incomplete query
   boundaries, malformed schema, conflicting delimiters, invalid escapes,
   truncated strings, and all other parse failures remain explicit pending
   omissions. No arbitrary recovery or silent source mutation is allowed.

The policy is a bounded syntax-repair allowlist, not a general JSON repair
heuristic. The crisp mechanism is an exact byte/character-preserving recognizer,
an immutable derived manifest, and existing shared renderer/admission checks.

The policy decision is this short source-integrity rule. The crisp helper is
just the shared serving validator, exact receipt hashing, and durable omission
write; it does not choose or infer a replacement argument.
