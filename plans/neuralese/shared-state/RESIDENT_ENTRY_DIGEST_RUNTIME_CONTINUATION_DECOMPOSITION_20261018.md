# Resident Entry Digest Runtime Continuation — Decomposition

## Policy and boundary

This continuation covers one metadata-only runtime update: the approved per-entry JSON sequence digest implementation replaces full-list canonical JSON allocation in resident validation. It preserves the exact old digest bytes and hashes, changes no tokenizer/parser component, data source, tensor operation, optimizer value, RNG state, completed exposure, or training source. The existing `native_capacity_continuation` metadata review/certificate mechanism remains the policy boundary. Its `METADATA_CHANGES` inventory now admits the shared `ordered_json_sequence.py` helper and the continuation module itself; the independently reviewed old/new runtime inventories and explicit `changes` bind the actual `native_resident.py` and helper bytes.

## Steps

1. **Bind the old runtime and measurement.** Read the old runtime inventory, source geometry certificate, and its actual transitive numerical/capacity evidence through the existing continuation verifier. Crisp code is required for exact SHA bindings, arithmetic maxima, and immutable evidence paths.
2. **Check the reviewed code delta.** Require the full old/new package inventories and derive their changed-file set exactly. The review must explicitly bind the actual old/new `native_resident.py`, helper addition, and any change to `native_capacity_continuation.py`; require the seven parser/source-policy components to remain byte-identical. The existing metadata verifier rejects any changed file outside its reviewed allowlist. This is deterministic artifact verification, so it is crisp.
3. **Check preserved training state.** Require the existing metadata-continuation conditions: optimizer state unchanged, explicit metadata memory allowance and measured headroom, unchanged geometry identity, and the same parent checkpoint. The owner-level intent is recorded in the review; the verifier checks its exact facts.
4. **Issue a derived geometry continuation.** Reuse the source certificate’s limits and evidence under the new complete runtime digest, with `new_parent_measured: false`. This does not claim a new model measurement.
5. **Bootstrap only the private clone.** Derive a new clone plan/current generation and request from the original bound current state and prepared QA append, replacing only runtime inventory, geometry-certificate binding, predecessor generation, and consequent content hashes. Preserve original current, request, source, geometry proof, optimizer state, and exposure as immutable evidence.
6. **Run the unchanged resident consumer.** The standard `ResidentAdoption.poll` validates the derived request using the exact new overlay runtime. The clone is private; no live pointer or model state is published.

## Proof and recovery

Tests must show these helper/runtime files pass through the existing metadata review, parser/source-policy changes fail, unreviewed file changes fail, and the derived certificate retains the old measured origin and limits. The remote clone receipt records old/new code inventories, certificate and request bindings, current-generation predecessor, resource telemetry, and a live-pointer before/after check. If the poll fails, retain the original request and its failure receipt, then repair the specific guard; never rewrite an old request or claim it ran under code it did not bind.
