---
description: Resolve the finite source-authorized work frontier while retaining each actual window delta.
args:
  request: TurnRequest
  source: TurnSource
  captured: CapturedTurn
returns: WindowState
---
1. Build WindowState from the actual CapturedTurn: copy the immutable initial board into current board,
   retain all original announcements/reference bindings, and copy its work slots and remaining IDs.
   Keep source and request unchanged. Initialize queuedTriggers and windows empty.
2. Combine unresolved source, timing and capture findings. A missing decisive timing or identity premise
   makes the state finished with that unresolved reason before any unsupported effect is committed.
3. Invoke advanceWindow.iterateOn with this state and measure remainingWorkSlotIds.length. Its finished
   predicate reads state.finished: no active permitted work remains, or an explicit unresolved obligation
   prevents further progress. Inactive one-use trigger slots may remain without active work.
4. Every progressing child result must consume exactly one originally inventoried active work slot and
   preserve earlier windows. Activating a previously inventoried trigger leaves the total remaining-slot
   universe unchanged. Source-defined capacities, not a chosen turn count, establish termination.
5. Consume the completed child's actual board, queued decisions, trace and unresolved fields. Verify
   there is no active unconsumed slot when claiming a completed determinate turn. A failed transition
   remains a failed contribution and is repaired rather than replaced by an invented next board.
6. Return the complete WindowState. Audit will inspect the actual per-window snapshots and selected
   effects; it cannot reconstruct an apparently correct final state independently of this trace.
