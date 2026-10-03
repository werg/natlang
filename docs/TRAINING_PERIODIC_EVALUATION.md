# Training evaluation sidecars

## Periodic held-out loss

`train_lora.py` can optionally evaluate a fixed, deterministic hash-ranked subset of the held-out split. Enable it with `--periodic-heldout-every N`; the subset defaults to 128 examples and can be changed with `--periodic-heldout-count N`. Selection is by SHA-256 rank of `(seed, row ID)`, independent of source-file order. The selected row IDs are hashed into each metric record.

For a one-time measurement of an existing checkpoint before its next interval, resume with `--periodic-heldout-at-start` as well. This uses the same step/subset/weight identity and does not duplicate an already completed measurement.

Metrics append to `OUT/heldout-periodic.jsonl`. Each completed record contains checkpoint step, data and split identities, model/revision identity, trainable-weight digest, token-weighted masked assistant loss, supervised token count, evaluated/skipped example counts, and class/source/source-group support. The nested numeric `metrics` object is intended for observers such as the W&B sidecar. A metric is written only after the corresponding optimizer checkpoint is durable. On resume, a due but missing evaluation runs before another training step; an already-recorded evaluation is skipped only if the checkpoint's trainable-weight digest matches. A partial final JSONL line is preserved in a recovery sidecar and discarded from the active log so that the due evaluation can be repeated. Complete malformed records and identity mismatches fail closed.

Periodic evaluation switches the model to eval mode, disables gradients, restores its prior mode, and restores Python, CPU Torch, and CUDA RNG states. It does not call optimizer or scheduler methods. The ordinary baseline and final held-out loss behavior remains unchanged; those legacy metrics are example-mean over at most the first 100 held-out rows. Periodic loss is token-weighted over its separate hash-ranked subset, so the two series should be labeled separately.

## Proposed execution-evaluation snapshot sidecar

This remains a separate, future operator workflow; the training loop does not start a server or call a provider for execution evaluation.

1. At an optimizer checkpoint boundary, retain the adapter snapshot and state already produced by `--snapshot-every`. A sidecar manifest should pin the step, trainable-weight digest, checkpoint-state hash, base-model revision, tokenizer/template/runtime identity, and snapshot file hashes.
2. Pin a small validation-only turn manifest from the existing held-out execution corpus. Use the existing paired turn evaluator's per-cell deterministic sampling and exact row digests. The manifest must prove disjointness from training IDs/source groups and contain no training targets selected after inspecting candidate output.
3. Run the evaluator only after the trainer has stopped cleanly at that checkpoint, on a separate local inference endpoint and isolated port. Use deterministic decoding and fixed per-call token/turn budgets. Do not hot-swap weights in the training process or share its optimizer process.
4. Write immutable results beside the snapshot, with exact/right-tool scores by evaluation cell and per-row outputs. Bind the result to both snapshot and manifest hashes. Preserve prior step results; never overwrite a completed evaluation.
5. Treat the sidecar as observation for human review. It does not change optimizer state, training examples, checkpoint selection, or publication automatically. Resume the trainer from its exact checkpoint after the sidecar exits.

The execution sidecar should be added only after a concrete runner can verify these pins, ensure the training service is paused, and clean up its inference server without touching the training checkpoint. No execution server or model call was started as part of this change.
