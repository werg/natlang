---
description: Review whether emergency egress stays independent of every puzzle state and prop failure.
args: { room: EscapeRoomState; frontier: PuzzleFrontierAnalysis; sourceEvidence: EvidenceItem[] }
returns: ExitAvailabilityReview
---
Check the declared room exit invariant against current venue, fire/life-safety and staff-intervention evidence. The graph may not make exit require a solved node, powered prop, participant action or staff hint. Consider the initial state, every frontier state, missing-prop branches and terminal soft-lock states. Return verified only when the supplied current evidence supports an immediately usable, independent egress path across all of them. An authored true flag without current source evidence stays unverified; preserve the source IDs and exact conditions still needing inspection.
