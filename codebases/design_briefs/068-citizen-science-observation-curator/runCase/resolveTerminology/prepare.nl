---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FieldObservation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for resolveTerminology. Interpret local names and descriptive phrases against project terminology and context. Retain multiple plausible identities when evidence is insufficient, and avoid forcing unusual observations into familiar categories merely to complete the dataset. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
