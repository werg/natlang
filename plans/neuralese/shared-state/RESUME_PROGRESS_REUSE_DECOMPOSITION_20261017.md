# Reuse collected continuation progress

Actual cloud startup stacks show the trainer collecting and deeply copying the
same historical stage frames twice before the first preference update. The
supervision policy remains: preserve exact optimizer, RNG and exposure while
returning the GPU to useful work promptly.

1. Collect independent historical progress once from the loaded training state.
2. If restoration uses that same state object, reuse the collected history as
   read-only input. Otherwise retain the existing collection path.
3. Copy the selected active frame before modifying it, and restore optimizer,
   FP8 state, RNG and global step from the current checkpoint exactly as before.
4. An absent stage still returns no restored frame; preserve all other stage
   obligations for subsequent checkpoints and scheduling.

This is crisp ownership and allocation plumbing in the existing shared
continuation mechanism. It changes no numerical operations, model-facing text,
recipe policy, checkpoint identity or training credits.

## Release decoded setup metadata before checks (2026-10-19)

After exact optimizer/FP8/RNG restoration, retain only the prior policy-capacity
proof, pending admission record and completed foundation identifiers needed by
subsequent setup. Release all obsolete decoded/restored predecessor aliases
before gradient and capacity checks, rather than after them. Live sources,
optimizer state, counters, checkpoint options and carried stage progress keep
their existing ownership; no predecessor map is mutated to recover memory.

A real interrupted/resumed tiny-model run tracks the decoded training dictionary
with a weak reference, verifies collection before the restored-model check, then
performs the remaining update and checks exact exposure and monotonic steps.
Existing continuation tests compare resumed weights, optimizer and RNG exactly.
This reduces setup lifetime; actual cloud memory improvement remains to measure.
