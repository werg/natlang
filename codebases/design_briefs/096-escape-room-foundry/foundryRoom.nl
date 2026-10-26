---
description: Design, verify, playtest and operationalize a safe escape-room graph.
args:{ sourceRecord: EscapeRoomProject; sourceEvidence: EvidenceItem[];room:EscapeRoomState;change:string[];proof:PuzzleProof}
returns:EscapeOperationsPacket
---
Read `creativeBrief`, `venueId`, `propIds`, `designerId` and `targetDurationMinutes` from the top level of the EscapeRoomProject record. Carry its accessibility route and puzzle prerequisites into the graph; do not replace them with a default room skeleton. Carry every source fact and evidence ref into the analysis branch that depends on it; when any transformed proposal conflicts with this anchor, hold that branch for review instead of accepting the placeholder.

Bind constraints = `modelRoomConstraints(room.brief, room.props, change, sourceRecord, sourceEvidence)`. Fictional stakes do not suspend actual exit, accessibility or staff intervention. Require the supplied `PuzzleProof` from the deterministic `verifyPuzzleGraph` service to match the exact room revision; if required goals are unreachable, a cycle creates a lock, or emergency exit invariant fails, return blocked before authoring clue flavor.

Bind clueReview = `auditClueFairness(room, proof)`. Check that information is discoverable before each required action, that plausible interpretations do not create underdetermined answers, and that out-of-order discovery leaves the world coherent. Bind variants = `designAccessibleRoutes(room, constraints, clueReview)`; every sensory opt-out must preserve an equivalent clue path and the same required goal.

Bind failurePlaytests = `rehearseFailureStates(room, proof, variants)`. Simulate repeated actions and a missing/broken prop; each reachable failure requires a staff recovery route with no participant trapped. Bind operations = `packageSessionOperations(room, proof, variants, failurePlaytests)`. Include setup, reset, hint ladders, prop inventory and pre-session state verification. A reset is not verified until the post-reset state matches the declared baseline and session IDs are isolated. Return for separate safety and enjoyment review.
