# Search/OWL over-cap port-record graph adapter

## Natlang-native policy

The adapter is a typed natlang transformation from each admitted
`natlang.port-record/1` source row into one train/test graph. Its source identity,
task, writer instructions, typed result, complete source list and exact references,
writer context, consumer context, withheld fields, target and alternatives,
outcome, lineage, licence facts, split, and split groups remain attached to the
derived row. A declared static teaching view supplies the source text as the
producer's exported span; the consumer imports that span where the withheld
source belongs and receives the original consumer request and target. This is
an explicit transformation of authored record structure, not a claim that an
actor captured this state or that the source was an observed model-generated
message. Consumer imitation is separate from source admission: `gold` and
`checked` outcomes follow the S1 target contract. A `teacher`/`repair`/`failed`
label does not become a positive merely because its task belongs to a broadly
approved program. A separate reviewer may promote an otherwise non-positive
target only with an exact raw source-row hash, exact target hash, decision ID,
and immutable proof hash. Unpromoted targets remain in the consumer context
with zero target weight, while the producer span remains independently useful.
Receipts report consumer-positive graphs/tokens separately from producer
supervision so source-only exposure cannot be mistaken for consumer learning.

## Stages

1. **Read and validate** — stream the selected plain JSONL port-record file
   through the shared strict `read_record_rows` schema contract. Bind the
   selected file's
   SHA, source corpus receipt, current protection snapshot, and renderer/code
   identity before producing output. Preserve every matching source row,
   including repeated IDs with distinct content; exact duplicate content may
   share a deterministic content identity while the receipt keeps every source
   occurrence.
2. **Normalize structure** — reuse the shared port-record renderer's typed
   writer/consumer semantics. Render writer instruction/context and each full
   source in its original order. Render consumer history and request with the
   producer payload omitted exactly when `consumer.withheld` says `sources`;
   otherwise preserve the visible source context. Retain exact references and
   all non-target metadata as provenance. Do not truncate source, history, or
   target text. Oversized model inputs are paged only at complete message/source
   boundaries and carry a source-bound continuation receipt; no row disappears
   because a conversion cap was reached.
3. **Build the native graph** — use shared tokenizer provisioning and the
   current native role-token IDs for both backbones. Create a writer invocation
   with an explicit producing span over the statically supplied source payload,
   then a consumer invocation importing that exact span at the typed source
   position. Preserve each invocation's causal prefix, consumer turns, target,
   and aligned role weights. The adapter never discovers edges by searching
   marker-like text. Use the existing `native_graph.parse_graph` contract as
   the final graph validator.
4. **Qualify identity and split closure** — derive collision-safe graph IDs
   from source ID plus canonical full source-row content identity. Preserve the
   original ID and row offset/SHA in the crosswalk. Bind all source groups;
   require train and held group sets to be disjoint and compare them against the
   latest held, quality, and decision-protection closure. Do not turn an
   unsuccessful or unverified outcome into a positive answer; retain the full
   target as zero-weight context unless an exact reviewed-target receipt
   authorizes that occurrence. Load the current sorted
   `training/decision-holds/*.jsonl` through the shared decision-hold loader
   and bind every hold-file SHA. A source/trajectory ID hit routes the full
   graph to held data; an exact held target-decision ID or source-row SHA zeros
   only that consumer target while preserving full context. Record exact
   matches and counts; zero hits means the binding was checked across every
   selected row, not assumed stale.
5. **Write** — stream directly into atomic compressed train and held outputs.
   The current 7,027-row selected input is rebuilt from its immutable source
   after interruption; output is only published when both compressed outputs,
   omission ledger, and receipt are complete. Keep memory bounded by one source
   row plus graph and tokenizer state; compressor and file handles close on
   every exception. A resumable per-row frontier is future work, not a current
   capability claim.
6. **Admit** — independently verify both tokenizer views, exact twin/source
   membership, target masks, native imports/exports, all source occurrence
   accounting, complete protected-group closure, output hashes, and omission
   reasons. Only then register immutable corpus manifests and stage the same
   shared adapter output for each model.

## Crisp mechanisms and rationale

- Python is limited to JSONL/zstd I/O, tokenization, exact role-token insertion,
  atomic writes, hashing, offset frontiers, and schema/protection verification:
  these cross the model/filesystem boundary and require exact behavior.
- The semantic choice that source text is the static writer payload, that
  withheld material is omitted from the consumer's own tokens, and that
  unsuccessful outcomes never become positive targets is expressed here as a
  named natlang transformation policy and recorded in each output receipt.
- The model-specific tokenizer and marker IDs are inputs; graph structure,
  source policy, split handling, and validation are shared across both models.
- No external generation, model call, arbitrary length cap, or silent
  whole-record omission is part of the adapter.

## Evidence required before admission

- Strict source parse count equals the 7,027 source occurrences selected from
  the 7,002 omission rows (7,001 unique IDs), with distinct-content identity
  rebasing recorded for reused IDs.
- Every source occurrence has exactly one output or an explicit, source-bound
  omission reason; train/held targets and groups are disjoint under current
  protections.
- Both model outputs pass native graph parsing and graph-edge/weight checks;
  matching source occurrences have equivalent model-neutral provenance even
  when token sequences differ.
- The admitted Search/OWL program-group index is a source-group join, not an
  occurrence-level target approval. Any promotion of old port-record targets
  must additionally bind the exact source row and decision/action evidence.
- A bounded fixture covers withheld and visible sources, multiple writer and
  consumer turns, alternatives/exact references, failed outcomes, repeated ID
  variants, train/held grouping, paging boundaries, interrupted resume, and
  corrupt-cache fallback.
