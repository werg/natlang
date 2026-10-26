---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for normalizeObservations. Parse field notes and structured submissions while preserving original wording, location precision, and observation method. Distinguish what was directly seen from inferred identity or cause, and flag missing metadata without inventing exact coordinates or times. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
