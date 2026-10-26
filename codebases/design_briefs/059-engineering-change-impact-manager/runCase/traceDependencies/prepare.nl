---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ChangeApplicability[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for traceDependencies. Traverse bills of material, interfaces, process instructions, and supplier commitments through exact dependency graphs. Judge whether each indirect dependency is materially affected and retain uncertain impacts for engineering review instead of broad automatic replacement. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
