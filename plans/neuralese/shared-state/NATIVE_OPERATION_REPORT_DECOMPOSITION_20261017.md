# Keep operation control small while preserving exact row evidence

The recipe controls finite work and retains every source obligation. Control
messages must not duplicate a million-row ledger at every durable state write.

1. Keep the complete optimizer/RNG, row bindings, source receipts, visit counts
   and stage histories in the existing full checkpoint. Never summarize training
   state or replace row credits with aggregate credits.
2. Reports retain real stage totals/completion and exact source identities.
   Repeated source-group catalogues and missing-row maps become explicit hashed
   detail references into that same checkpoint. Group exposure remains exactly
   recoverable from its source index and saved row counts.
3. A crisp verifier resolves a requested detail using the existing checkpoint
   loader and source parser, checks its complete content hash/count, and rejects
   changed checkpoint/source/detail identities. No global tensor cache is added.
4. Controller state retains all unknown control fields and only replaces known
   bulky report payloads with an exact report binding. Foundation completeness,
   source obligations, cadence, checkpoint identity and future guards survive.
5. Existing readiness and semantic consumers retain their actual row-level
   foundation and communication evidence. Ordinary control does not resolve
   detail maps. Full row data is fetched only when its actual consumer asks.
6. Per-stage scheduling may reuse the existing trusted trainer report/readiness
   pair only at the actual current checkpoint SHA, with exact report hash, native
   identity and equal stage summaries. Absent/stale receipts use the full saved
   checkpoint; changed bound evidence rejects. No persistent cache is introduced.

This is serialization, exact verification and durability plumbing. It changes no
model-facing behavior, arithmetic, masks, optimizer, source policy or exposure.

The first component applies only to controller records and current-progress
reuse. Existing trainer report/readiness bytes remain unchanged. The shared
checkpoint-detail functions are ready for the separately reviewed trainer
serialization followup; that followup is required to shrink worker reports too.
