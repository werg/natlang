# Pinned cooperation, research and raw multi-agent sessions

The natlang training policy preserves every usable original session and its
actual causal context, actor, shared task and reported outcome. Existing source
quality, protection and whole-trajectory target policy remain authoritative;
conversion does not invent an execution oracle or grant training admission.

1. **Pinned source reader (crisp exact I/O).** Reuse one receipt-bound iterator
   for Parquet, JSONL (plain/gzip/zstd) and whole JSON (plain/gzip). Expose the
   supported-file predicate and integer original row indices to the export
   driver. Preserve blank-line JSONL indices; whole-object JSON has row zero,
   and top-level JSON arrays retain original element indices. Verify exact file
   size/digest and receipt membership before reading.
2. **D14 original cooperation contexts (exact field mapping).** Preserve each
   agent's original solver and summarizer/compaction contexts as separate
   complete sessions. Never concatenate incompatible compaction contexts into
   an invented history, or keep only the final compacted transcript. Retain the
   original final messages unless they are an exact already-exported segment.
   All sibling contexts share the exact original task group, retain original
   path/agent/segment/kind/model/config/status provenance and explicit failures.
3. **D32 structured research channels (exact normalization).** Preserve every
   original text, reasoning channel, system/tool definition and observation.
   Preserve recipient/name/channel facts. Map genuine explicit tool dispatches
   through the existing call representation; retain other structured content
   textually rather than dropping it or fabricating a call/result. Never train
   an empty message because the source used content blocks instead of strings.
   Chunk/session siblings retain their shared original question identity.
4. **D19 original raw traces (exact event mapping).** Prefer the pinned full
   raw traces to reordered Parquet summaries. Preserve real agent/episode
   contexts, calls, observations, communication, feedback and reflection in
   source order. Shared task/experiment/run grouping binds sibling agents and
   episodes. Unsupported structures become explicit omissions with raw source
   retained; known failed upstream outcomes remain non-positive.
5. **Existing trajectory/native conversion (shared mechanisms).** Use the
   existing Trajectory, full-session exporter, gold renderer, protection and
   native graph converter for both student tokenizers. No parallel renderer or
   source admission policy is introduced. Validate format adversaries and real
   pinned samples, including all compaction contexts and structured observations.

Every crisp component here preserves exact source facts or verifies transport;
none chooses training targets, teacher quality, promotion or a model policy.

The existing `agents.harmony_messages` is the sole structured research parser.
It now retains the original system content and each reasoning message, rather
than dropping the system block and folding reasoning turns into later calls.
`_harmony_text` keeps typed observations and non-text system configuration as
JSON when no verbatim text representation exists. Tool calls use original
recipient/argument bytes with deterministic position-derived plumbing IDs;
observations bind the preceding dispatch. The old OpenResearcher adapter reuses
this same complete mechanism. Generic OpenAI normalization also retains
structured content instead of turning it into an empty message.

D19 raw schema1.0.0 includes complete per-agent contexts and accepted usage
records. Sparse reasoning is assigned only when the exact current task prompt,
accepted response inventory and content/tool-call shapes agree. Usage
round/attempt coordinates are one-based; raw trace coordinates are zero-based.
Every assigned trace records its original usage/trace/message indices. Conflicting,
ambiguous or unsupported joins retain the unchanged session and original traces
with an explicit incomplete-alignment receipt, never a guessed target.

Local qualification, 2026-10-16:
- Shared reader: 25 tests passed; structured/trajectory regression suite:
  39 tests passed, with malformed-context continuation, exact failure labels,
  forged reasoning-shape rejection and structured observation preservation.
- Actual pinned samples verified against remote receipt file size/SHA: D14 four
  solver/summarizer/final contexts (47/46/10/11 original entries, including one
  terminal exit event retained separately); D32 332 messages, 110 reasoning spans,
  109 calls and 109 observations; D19 20 original agent histories, 129 exact
  reasoning joins, eight teacher candidates and 12 explicit failures.
- Both actual pinned tokenizers rendered the complete selected positive sessions
  through the shared full-session native converter: 24/111/13 graphs per port
  for D14/D32/D19, no truncation and unchanged source-group sets. Final ledger
  unit `acquired-structured-real-tokenizers-final-20261016` passed in 34.8 CPU s,
  577 MB peak host memory. Detailed source/tokenizer bindings are in
  `/home/werg/data/dgxops/acquired-structured-formats-proof-20261016.json`.
- Tracked-file quick check passed. Admission, complete source conversion,
  protected split closure and actual training exposure remain separate steps;
  this format proof does not claim task success or locally verified source quality.

Independent review correction: named legacy `function` feedback and assistant
`function_call` use the shared OpenAI normalizer, preserving exact call arguments
and observed response text. Named feedback cannot become a dispatch merely
because its JSON result contains `name` and `arguments`. For D19, an expected
negative verdict alone is not actor failure: a correct negative evaluation can
be useful teacher behavior. Failure classification uses reported incorrect
verdict, nonpositive reward or actual failed task-check results; expectation is
retained as a separate source fact, never a local oracle.
