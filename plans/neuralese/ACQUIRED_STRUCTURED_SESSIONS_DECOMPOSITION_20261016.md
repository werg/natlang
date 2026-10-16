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
