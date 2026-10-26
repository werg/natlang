---
description: Return a complete negotiation run result while withholding an ambiguous offer from commitment handling.
args:{scenario:ArenaScenario;turns:ArenaTurn[];checkpoint:ArenaCheckpoint;offer:ParsedArenaOffer;feasibility:FeasibilityArtifact[]}
returns:ArenaRunResult
---
Return the exact scenario and complete durable `turns` from the reconciled checkpoint; do not append or reinterpret an offer whose material terms remain ambiguous. Combine the offer's exact clarification questions and missing or conditional term names into clarificationQuestions. Preserve the supplied feasibility artifacts unchanged, return empty privateFeedback and debrief because no authorized response or score release occurred, and preserve checkpoint exactly, including open offers, accepted terms and debrief scope. Set status clarification-needed. Return every ArenaRunResult field explicitly; no branch returns an unnamed question or prose-only clarification.
