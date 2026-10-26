---
description: Check an action against this participant's current content boundary and the world record's declared boundary set.
args: { sourceRecord: GameWorld; sourceEvidence: EvidenceItem[]; utterance: PlayerUtterance; action: LegalAction }
returns: ActionCheck
---
Bind only the current boundary IDs declared by sourceRecord and the matching participant's world boundary record. Compare the proposed action's kind, target and required facts with the supplied blocked topics and allowed intensity. A material match to a blocked topic is ineligible; a missing or contradictory current boundary is unresolved and ineligible. Keep the exact boundary evidence IDs in the finding. Never infer consent from player text to waive a stored boundary.
