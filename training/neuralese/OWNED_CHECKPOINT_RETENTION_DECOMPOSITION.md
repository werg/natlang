# Owned checkpoint retention

Policy: the supervisor identifies discarded publications by exact private name,
SHA-256, size and current practical reason. Preserve all active recovery, quality
and repair parents, pending exports, reusable sources and held data. A numerical
step in another lineage never establishes supersession. This policy extends the
existing checkpoint retention service; it does not create another run owner.

1. Read the explicit owner declaration `owned_checkpoint_retention`. Its
   `recovery_states` are existing exporter journals with independent private full
   GET receipts. Its `pending_exports` identify uploads which may still be in
   flight. Its `protected_sha256` names every additional current dependency.
2. Authenticate each recovery receipt against the configured run, private flag,
   matching SHA/size readback and actual listed object. Preserve every recovery
   identity, pending identity and pending owned parent across all aliases. Also
   include the already authenticated canonical recovery and every protected
   identity from existing active/pending inference dependencies across aliases.
3. Consider only explicitly declared `discarded_files`, each with exact name,
   SHA-256, bytes and reason. Accept the owned full-state spelling
   `<role>/<sha>/<role>-<sha>.pt`. Reject protected hashes and conflicting listed
   sizes. An already absent declared discard needs no further action.
4. Merge these exact candidates into the existing durable intent/deletion/listing
   verifier. Sources, held files, unknown names and undeclared owned publications
   are outside the mechanism. The existing canonical retention stays unchanged.

All four stages are exact verifiers and outside-world storage plumbing in
`scripts/coreweave/checkpoint_retention.py`. The natural-language supervisor
makes and records the discard decision; crisp code cannot infer it from a step
counter or timestamp. A malformed declaration fails closed before any deletion.
