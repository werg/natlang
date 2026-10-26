---
description: Build a room-revision-bound puzzle proof from graph structure, every solution branch and an independent egress review.
args: { room: EscapeRoomState; sourceEvidence: EvidenceItem[] }
returns: PuzzleVerification
---
1. Bind structure = `validatePuzzleStructure(room)`. If its roomId or revision do not match this room, or it reports duplicate node IDs, invalid references or prop coverage gaps, preserve those exact failures in the proof. Do not run reachability against ambiguous node identities.
2. Bind frontier = `explorePuzzleFrontier(room, structure)` using every distinct solved/available state. Its returned state IDs, reachable nodes, unreachable goals and terminal soft locks are the graph's result; do not replace them with a supplied proof or a preferred solution path.
3. Bind cycles = `tracePrerequisiteCycles(room, frontier)`. Preserve actual cycle members, blocked goals and affected terminal states separately; do not assume every cycle makes every goal impossible.
4. Bind egress = `verifyExitAvailability(room, frontier, sourceEvidence)`. exitInvariantHolds is true only when this review is verified and the room's declared emergency exit is available. The design flag alone does not establish real egress.
5. Construct PuzzleVerification with graphRevision = room.revision; reachableNodeIds, unreachableGoalIds and softLockStates from frontier; cycleNodeIds from cycles; propCoverageGaps, invalidReferences and duplicateNodeIds from structure; exitInvariantHolds from egress; and all explored states. Return every field, including empty findings, explicitly. Empty findings mean the named check found none; they do not establish checks this operation did not run.
