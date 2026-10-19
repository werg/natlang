# Stale RAW collection retirement

## Policy

The RAW round remains incomplete when its learner or bridge has stopped with an
unconsumed pending round. The supervising owner decides when to replace the
stale operation. Retirement records that decision and leaves all source,
selection, rollout, learner, and replay files intact. The next scheduler pass
returns to mandatory SFT obligations first, then collects a fresh attempt for
the same RAW round. No round, task, optimizer update, or exposure is credited by
retirement.

## Exact mechanism

1. Bind the exact model config, production state, cycle cursor, selection,
   operation directories, the exact parked legacy-provider receipt, and a same-host process census. The old worker never
   recorded an owner identity, so the authority states that fact explicitly;
   it does not invent historical PID/birth evidence. It also binds the exact
   distinct successor entry which is allowed to remain live.
2. Check all captured process identities and command lines against the live
   process table, then repeat a current `/proc` stale-path scan under both
   controller leases immediately before pointer publication. The provider
   receipt must be recent and show the exact four legacy provider/heartbeat
   units inactive, with `learner_checkpoint_mutations=0` and no training credit. Any live process
   referencing the retired config, updater, bridge or state/cursor blocks the
   transition. Only the exact successor PID/birth/boot and wrapper command is
   exempt.
3. Verify the canonical controller lease either through this process's inherited
   descriptor or through the exact permitted live entry owner's `/proc/<pid>/fd`
   and `fdinfo` descriptor. An external operator leaves that flock held and takes
   only the separate production-state lease; it never substitutes its own lease.
   Under the canonical owner lease and the production lease,
   snapshot the old JSON bytes and build two exact candidates. The production
   candidate keeps the same round, offset, trained-round count, selection,
   checkpoint, and debts; it records all retired operation fields and increments
   only `collection_attempt`. The cursor records the retirement receipt, removes
   only the pending-round claim, and moves to `phase=sft,next_stage=0`.
4. Persist intent and candidate bindings before replacing either pointer. On
   restart, publish only the exact candidates or accept their exact already
   published bytes; snapshot bytes and hashes must match the original authority.
   Once a completed receipt exists, a retry authenticates its retained intent
   and snapshots without requiring a live owner or a newly fresh quiescence
   sample. Unknown pointer successors fail closed. The final receipt
   explicitly says no RAW round or training work was completed.
5. The existing scheduler uses the incremented collection attempt to select a
   new collection output directory. A shared cycle-work helper places all new
   SFT and bridge jobs under a fresh `retired-attempt-N` namespace, while old
   request and replay directories remain untouched. The existing selection
   snapshot and its order remain unchanged.

The natural-language owner policy decides that a stale operation should be
retired. The Python helper is limited to process identity checks, filesystem
bindings, locks, atomic snapshots, and exact pointer publication.
