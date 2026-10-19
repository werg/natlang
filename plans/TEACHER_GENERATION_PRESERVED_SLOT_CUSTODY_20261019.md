# Preserved teacher-generation slot custody

## Natural-language policy

At a generation-runtime handover, preserve every already-started batch exactly. For each expected batch, the supervisor checks whether its same feeder is still live or whether that same feeder's completed batch has an authenticated boundary receipt. A process exit alone never means completion. A sealed directory alone never identifies its owner or runtime. A missing slot, substituted owner, changed case/queue/source bytes, stale selection intent, or mismatched task-runtime epoch blocks the handover.

## Exact mechanism

1. The handover manifest supplies the exact batch, original process identity, three sealed source bindings, and the expected `source_sequence.started` receipt binding.
2. The shared guardian resolves that batch only from the actual active/ahead slot state or its authenticated closed source-sequence receipt. It verifies the same owner identity and immutable start receipt.
3. The operator makes one call to `verify_preserved_primary_slots` with its manifest. The manifest supplies the complete expected slot list, the source-sequence `started` map, and captured `primary_prefetch`, completed-ahead, active-ahead, and primary-launch slot snapshots.
4. The shared verifier checks the source-cases, queue and teacher-source files against the handover bindings, then checks the started receipt against the same sealed files and selection intent.
5. It verifies that the slot, selected prefetch intent and sealed teacher-source name the same task-runtime epoch.
6. A live owner returns a live-custody result. A dead owner is accepted only if the exact guardian boundary is bound, complete, and names the same batch, queue, teacher source and journal/score bindings. The finished-ahead slot and its dead owner identity must still match.
7. The same read-only call runs before handover and after guardian replacement. It does not adopt, launch, archive, or relabel batches.

## Decomposition

- Batch/runtime policy remains in the existing natural-language supervisor and source-sequence stages.
- `verify_preserved_primary_slot` is crisp because process identity, hashes, exact path binding, and durable boundary facts require exact verification.
- The operator integration is one call to `verify_preserved_primary_slots` replacing duplicated per-slot checks; it does not add another scheduler or transition policy. It consumes the deployed v3 manifest directly, including its existing `bound_prefetch`, `bound_completed_ahead`, and `bound_active_ahead` fields. `bound_primary_launches` is optional; when absent, the original active batch is authorized by the manifest's existing `active_batch` and `original_owner`, then checked against its exact started receipt, actual slot owner, source seals, runtime, and any closed boundary.
- The deployed-manifest operator call is:
  ```python
  preserved = guardian.verify_preserved_primary_slots(root, control, state, manifest)
  ```
  The operator records `preserved` as its read-only custody observation before and after the handover. The direct manifest-shape regression covers the active predecessor plus batches 46 and 48 through their exact closed boundary receipts.
- Tests use synthetic exact receipts for live, completed-ahead, and closed-boundary routes and mutations for missing slots, arbitrary boundaries, changed owner, source bytes, cases, queue, and runtime.
