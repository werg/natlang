---
description: Run a versioned fictional negotiation while enforcing role authority, privacy, and offer timing.
args:{ sourceRecord: NegotiationExercise; sourceEvidence: EvidenceItem[];scenario:ArenaScenario;learnerUtterance:string;roleId:string;turns:ArenaTurn[];feasibility:FeasibilityArtifact[];checkpoint:ArenaCheckpoint}
returns:ArenaRunResult
---
Read `learningObjectives`, `participantIds`, `scenarioFamily`, `instructorId`, `conductPolicyId` and `workflowStatus` from the top level of the NegotiationExercise record. Use them to constrain the role-play and debrief; participant identity, instructor authority, and conduct policy cannot be synthesized from the learner utterance. Carry every source fact and evidence ref into the analysis branch that depends on it; when any transformed proposal conflicts with this anchor, hold that branch for review instead of accepting the placeholder.

Require checkpoint scenario revision to match. Bind parsed = `parseIssueLevelOffer(scenario, learnerUtterance, turns, sourceRecord, sourceEvidence)`. Keep missing or conditional terms explicit. If ambiguity could alter a trade or acceptance timing, return clarification questions and do not update commitments.

Bind legalReview = `checkOfferAuthorityAndFeasibility(scenario, roleId, parsed, feasibility)`. Exact feasibility artifacts must match the current scenario and offer ID; private utility bands remain hidden from the learner. Bind response = `chooseBoundedCounterpartAction(scenario, roleId, parsed, legalReview, turns)`, using only this role's authorized facts, priorities, reservation conditions and permitted tactics. Refuse any response outside authority or conduct limits.

Bind turnUpdate = `applyNegotiationTimingRules(scenario, parsed, response, turns)`. A withdrawal or acceptance applies only under the exact timing rule; an offer is not committed by exploratory wording. If instructor changes difficulty, open a new scenario revision and preserve this episode's hidden state. Bind debrief = `debriefObservedNegotiation(scenario, turnUpdate, legalReview)` only after authorized debrief policy allows private score release. Ground feedback in observable turns, exact feasibility and learning goals; do not reward coercion or invent motive.
