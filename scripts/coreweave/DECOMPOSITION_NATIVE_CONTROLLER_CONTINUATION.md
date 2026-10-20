# Drained supervised continuation

The recipe and its owning bootstrap choose the supervised scheduling quantum. This helper does not choose work or grant readiness. It verifies and durably records the mechanical continuation of an already authenticated drained recipe.

1. Ask the existing handoff verifier for the actual source plan and stage ID. This remains a crisp identity check; do not reconstruct exposure or optimizer state.
2. Verify the executing package and the existing capacity certificate using the existing resident verifiers.
3. Find that exact stage by ID and independently require its `native_state_sft` kind. IDs identify instances; kinds select implementations.
4. Apply the caller's positive declared quantum and the verified resident capacity binding. Preserve every other recipe field and stage.
5. Commit the immutable recipe and provenance proof through the existing durable JSON mechanism. Matching retries reuse their contents; conflicting retries fail.

Model-facing prompts and numerical execution are unchanged. Tests exercise the actual stage ID/kind distinction and preservation of the complete recipe.
