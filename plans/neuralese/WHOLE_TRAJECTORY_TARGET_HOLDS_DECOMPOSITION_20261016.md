# Explicit target holds in complete static trajectories

`--decision-exclude-ids` means deduplication: decisions already rendered in
another view still complete their whole trajectories. It is not a quality
hold. Keep that behavior and give explicit target holds a separate input.

1. Add `--target-exclude-ids` to shared/part rendering and bind the exact hold
   file hashes in resumable policy. Suppress held standalone targets. Preserve
   the full transcript, including held actions as causal recovery context.
2. For each maximal whole trajectory, map its held member records to exact
   assistant-turn ordinals from the source message prefixes. Require source
   and rendered assistant counts to agree. Emit source-hash-bound exclusions,
   the assistant count, and separate target versus context-only source IDs.
   Omit a document with no independently admitted decision target.
3. Native conversion verifies this mapping against actual parsed chat roles,
   assigns zero loss to those assistant spans, and retains all token IDs and
   other valid targets. Existing runtime-failure masking remains active.
4. Include target-mask semantics in whole-document deduplication and target
   statistics, so identical text cannot silently discard a different mask.
5. Whole-case/trajectory holds continue to use existing program exclusions.
   Benchmark/held-group protections remain whole-source exclusions. A
   target-only hold never overrides either. Text rows carrying these masks
   require the native conversion before training admission.
6. Verify real shared-renderer output plus both tokenizer/native conversion
   paths: excluded actions have zero weight, context is byte-identical,
   independently valid later actions stay supervised, dedup-only behavior is
   unchanged, and malformed mappings fail closed. No new prompt is introduced.

Whole rows use `source_record_ids` for positively admitted targets;
`source_records` retains the complete hash-bound source inventory, including
`context_only_source_record_ids`. Per-decision exclusions have explicit omission
records and do not imply that their context was removed from whole trajectories.

Independent review found and corrected two source-credit gaps before adoption:
every member now authenticates its exact target against the actual assistant,
including the final target; duplicate assistant ordinals and mixed split/group
membership fail closed. Positive source bindings must resolve to unheld,
non-opening, successful assistant spans. No positive source is credited merely
because its input prefix matches.

Validation: ledger `whole-target-holds-reviewfix-20261016` passed 56 focused
conversion, duplicate, part-merge and binding tests; quick checks passed in
`whole-target-holds-quick-20261016`, with final hardening checks in
`whole-target-holds-publishcheck-20261016`. The earlier final/final2 invocations failed
before collecting tests (wrong interpreter / guessed test path); neither is
counted as verification. Actual pinned LFM and Mellum tokenizers passed shared
render-to-native conversion in `whole-target-holds-real-tokenizers-v2-20261016`:
all transcript tokens retained, nine held assistant tokens masked for each,
seven/twelve positive target tokens respectively. This was a synthetic protocol
fixture, not generated or admitted training data. Evidence is in
`/home/werg/data/dgxops/whole-target-holds-real-tokenizers-20261016/validation.json`.
