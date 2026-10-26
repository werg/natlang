---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FacilityVisit[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for verifyResolution. Compare technician completion evidence and subsequent occupant observations with the original complaint. Distinguish a replaced component from resolution of the symptom, detect recurrence, and reopen only the relevant issue when work does not achieve the intended outcome. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
