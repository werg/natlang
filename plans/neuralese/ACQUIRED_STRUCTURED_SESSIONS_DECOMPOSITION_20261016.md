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

## D13 team cooperation and shared cooperation mechanics

1. Reuse the D14 context adapter for the D13 wrapper: original `messages`,
   `mini_swe_segments`, `mini_swe_info` and format facts map to the same complete
   context enumeration. Every actual agent/run/solver/summarizer/final context
   remains distinct. D14 prefers `_full_traj.json` when that representation
   exists, and otherwise retains the original `_traj.json` agent; a missing
   preferred copy cannot discard extra agents.
2. Verify adjacent evaluation/result/metadata/communication files against the
   same acquisition receipt before joining their facts. Record every dependency
   path/size/SHA in source provenance. Submitted is a finish status, not proof of
   successful tests. Actual failed upstream checks remain explicitly non-positive;
   source `verified`/held-out-test facts are upstream facts, not local certificates.
   Missing optional neighbors remain explicit unknowns, not invented success.
3. Keep original run/agent/feature identities and a common benchmark
   `cooper-task` repository/task group across D13, D14 and existing D17 views,
   alongside existing groups and original split facts. Agent aliases and different
   task features must not bypass held-source closure.
4. For source format `codex_stream_log`, upstream `[command]` messages combine
   the command and its observed output in one text block labeled assistant.
   Preserve the complete block as observed tool-log context with exact original
   position/role provenance; never invent a typed call/output boundary. This
   retains lower-weight whole-trajectory observation supervision rather than
   presenting returned file listings as generated assistant answers. Other
   original messages and genuine structured calls keep their existing handling.
5. Validate all five acquired D13 run variants and both agents with pinned samples,
   then the shared full-session converter on both actual production tokenizers.
   Bulk export/admission and comprehensive model exposure remain separate work.

D13 qualification (2026-10-16): all five run variants and both original agents
were loaded through the shared reader from exact pinned bytes, including all
four bound neighbor documents. The successful Codex variants retained 2+2
contexts and 108+115 messages (164 combined command observations); the sampled
unsuccessful MiniSWE variants retained 18+18+14 contexts, including every
7/9/5-segment compaction inventory, and remained nonpositive. Both actual pinned
tokenizers rendered all four successful complete agent sessions, preserving all
assistant targets and lower-weight observations. LFM lengths were
81,091/30,438/82,278/61,759 tokens; Mellum lengths were
89,893/32,595/93,328/66,248, with no truncation. Both rejected the actual failed
source from positive conversion. This proves format handling, not local task
success or GPU training of these long rows. Production scheduling must account
for their actual lengths.

The focused suite passed 44 tests. The actual pinned-byte/tokenizer ledger unit
`team-cooperation-both-agents-tokenizers-20261016` passed (7.4 CPU seconds,
686 MB peak host memory); detailed immutable file/tokenizer bindings are in
`/home/werg/data/dgxops/team-cooperation-source-proof-20261016.json`.
Export selection uses the shared `cooperation_trajectory_file` predicate.
Normalized export identity must include the acquisition/helper identity and
bound neighbor inputs: identical agent bytes across runs can have different
quality metadata. Full bulk conversion, cross-source held-group closure,
admission and actual training exposure remain separate operations.

## Remaining source formats: D12/D15/D16/D18

1. Treat acquisition and archive traversal as exact services: validate pinned
   file bytes, retain stable original member/row identities, never extract tar
   paths or execute source material. A shared archive reader yields original
   JSON documents with member provenance, and the cooperation context normalizer
   handles their agent contexts and receipt-bound neighbor facts. Prefer complete
   contexts over summaries; retain omitted raw formats explicitly for repair.
2. The D15 dataset is an upstream transformed two-agent view. Preserve every
   chronological original entry, actor, action, observation, reasoning and round
   transformation receipt. Expose only the released view, with no claim that
   unrecorded four-agent private contexts or native state are recoverable.
3. D16/D18 provide observed multi-agent logs with source failure annotations.
   Their natlang unit is an execution-diagnosis function: read the full observed
   trace and return the exact source-reported annotation object. Static rendering
   is crisp transcription, not a second quality policy or a generated repair.
   Preserve all raw trace text as input; never supervise failed logged decisions
   as positive agent actions. Null/missing labels and human-study taxonomy rounds
   remain distinct facts, not invented zero labels. Source judge output is teacher
   supervision, not an independently executed oracle.
4. Share exact MAST source/task/content groups across D16 and the MAST-derived
   D18 subset. Preserve source splits and leave-one-framework-out metadata;
   cross-source protected-group closure precedes admission. Injected anomalies
   retain their explicit synthetic provenance and task-family groups.
5. Use the existing normalized full-session/native converter for both pinned
   production tokenizers, current protections and decision holds. Export all
   complete normalized sessions and resumable graph shards locally under the
   memory ledger. Source conversion, actual admission and training exposure are
   separate receipts. No downloading duplicate sources or external generation.

Remaining-format qualification: 57 focused tests and the tracked-file quick
check passed. Complete source preflight found 5,025 D12 contexts (2,826 source
teacher candidates and 2,199 source-reported failures), 718 D15 actor views
(611 candidates and 107 failures), 1,661 D16 source-diagnosis sessions, and
2,220 D18 logs. The 83 MAST-clean `held_out` rows map explicitly to test, with
original split facts retained. Both the nullable labels and source human-study
rounds remain unchanged; execution-model identity is not misrepresented as the
unknown annotation-judge identity.

D12 source defects remain explicit: 19 Error records have no messages; one
original JSON member is empty/malformed; three raw Codex stream logs end before
the separately stored final response (including upstream disk-full recorder
errors). The exact-suffix verifier rejects claiming those contexts complete.
Non-session members and superseded summaries retain stable original archive
ordinal/size/digest accounting. A malformed or binary member does not discard
other usable members. No archive paths are extracted or executed.
