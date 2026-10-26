---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RecipientNeed[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for constructBundles. Assemble useful combinations rather than allocating isolated items that cannot serve the recipient’s purpose. Propose acceptable substitutions, identify missing components, and ensure the bundle’s condition and practical requirements match the confirmed need. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
