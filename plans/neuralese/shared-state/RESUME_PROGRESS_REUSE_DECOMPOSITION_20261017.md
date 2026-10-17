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
