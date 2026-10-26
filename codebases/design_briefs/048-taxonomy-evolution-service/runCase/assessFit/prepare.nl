---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for assessFit. Compare incoming records with category definitions and examples, not names alone. Distinguish a genuinely new concept from a poorly worded existing category, and retain ambiguous membership where the taxonomy permits multiple dimensions. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
