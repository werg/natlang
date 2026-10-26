---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: TaxonomyDefect[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for designCategoryRevision. Draft revised definitions, inclusion and exclusion rules, and parent-child relationships. Preserve stable category IDs where meaning remains stable, create new identities when meaning changes, and explicitly map split or merged historical categories. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
