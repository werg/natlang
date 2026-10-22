# Historical recipe reader replay

The supervisor keeps earned full states and current training moving while changing
runtime code. Historical recipe validation must authenticate the implementation
that originally enumerated that recipe's dependencies. A changed current reader
must never silently replace it, and changed bytes must not be treated as a path
relocation.

1. Read the proof's original package, complete code inventory and bound recipes.
2. Reconstruct recipe metadata in a fresh CPU process importing that exact
   package. Keep CUDA disabled. Check the package inventory before and after.
3. Check each recipe's actual path/hash; require the same ordered plan count.
4. Replay completed adaptations, drained handoffs and resident history using the
   relevant historical package. Continue applying the current exact lineage,
   source/held, checkpoint, optimizer and exposure verifiers.
5. Build future invocation plans using their declared future package. Never
   mutate old plans or grant training credit for replay.

The existing external-owner recipe reconstruction is the shared mechanism to
extend. The subprocess and file checks are crisp helpers for exact imports and
artifact identity; no model-facing text or training objective changes. Keep
byte-identical reader relocation checks strict. Test changed-reader histories,
package mutation, source/held mutation and bad subprocess output, and check the
actual fully loaded cloud control closures before replacing active training.
