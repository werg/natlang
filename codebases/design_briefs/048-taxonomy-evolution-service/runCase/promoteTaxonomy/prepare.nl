---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: TaxonomyImpact[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for promoteTaxonomy. Present impact, examples, and reporting consequences to the steward, then publish a versioned release and migration plan. Notify dependent systems, invalidate only relevant classification caches, and preserve old definitions for historical queries. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
