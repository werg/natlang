# Deferred in-process RAW owner handoff bridge

## Natural-language operation

The controller receives a deferred startup-import record and a completed
handoff from the same in-process owner. It verifies that the owner started from
the checkpoint selected by the startup authority, that the saved child
checkpoint records that exact checkpoint as its continuation parent, and that
the existing stale-round retirement still preserves the outstanding RAW debt.
It then publishes the child checkpoint as the current RAW parent and returns to
the already-required supervised recipe stage. The bridge grants no stage,
readiness, or RAW-round credit.

## Exact mechanism

1. Authenticate the deferred record, startup authority, dispatch launch,
   pre-signal process-tree intent, and original controller/config bindings.
2. Authenticate the completed drained handoff and full child checkpoint. Join
   its process-tree intent to the exact startup dispatch, and join the child
   checkpoint's continuation SHA to the startup-selected checkpoint SHA.
3. Authenticate the no-credit stale-pending retirement receipt and require the
   current state/cursor to be exactly its retired successor.
4. Under the existing controller and production leases, snapshot current
   state/cursor, durably write candidates that change only the current
   checkpoint references/readiness and a bridge receipt, then CAS both files.
5. Allow the ordinary runtime-handoff producer to run after the pointer commit;
   now its verified child equals the published current checkpoint, so the
   existing runtime handoff and `configure_model` path remain authoritative.

Exact hashes, byte bindings, process identities, locks, snapshots and atomic
JSON writes are crisp because they define the outside-world custody boundary.
No full startup checkpoint is reconstructed from the summary, and no model
policy or training credit is inferred by the bridge.

The later runtime-handoff proof accepts the bridge receipt as a distinct
boundary authority. It rechecks the bridge's current state/cursor candidates and
requires the existing mandatory supervised-stage inventory before treating the
cursor as SFT-first. This does not manufacture an `owned_operations` receipt:
the bridge reference remains explicit in the runtime-handoff proof, and the
first scheduled ordinary SFT work remains owed. The frozen startup runtime
package is checked against its original startup gate and code inventory; the
currently running verifier may be a later package revision and is not used to
relabel that startup package.

Runtime-handoff verification always reads the immutable config snapshot sealed
with the handoff, because applying the handoff rewrites the live config. The
scheduler clears the one-time bridge boundary only after it verifies that the
configured runtime proof names the same bridge request and checkpoint. It
records that applied handoff in a zero-credit consumed receipt while the owed
SFT work is pending, so later scheduler entries cannot mistake the original
boundary for new work. None of these steps changes debt or grants stage credit.
