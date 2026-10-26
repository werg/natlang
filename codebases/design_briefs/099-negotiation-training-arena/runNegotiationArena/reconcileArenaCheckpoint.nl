---
description: Reconcile the negotiation episode ledger after one timing-rule application.
args:{checkpoint:ArenaCheckpoint;scenario:ArenaScenario;turns:ArenaTurn[];currentOffer:ParsedArenaOffer}
returns:ArenaCheckpoint
---
Preserve the existing episode identity, difficultyRevision and debriefReleaseScope. Set scenarioId and revision to the current scenario only when this is the same episode revision; a changed scenario revision cannot reuse this checkpoint and must leave the run blocked for a new episode. Store the complete append-only turn list returned by the timing-rule operation. Recompute openOfferIds from offers whose turn state remains open and whose expiry/conditions have not closed them; do not retain an offer accepted, withdrawn or rejected by a later turn. Recompute acceptedTerms only from turns marked accepted under the scenario acceptance rule, preserving the exact issue-level terms and conditions. Never add exploratory, clarification-needed or conditional-but-unsatisfied terms to acceptedTerms. Return all ArenaCheckpoint fields, including unchanged private debrief scope, for exact resume.
