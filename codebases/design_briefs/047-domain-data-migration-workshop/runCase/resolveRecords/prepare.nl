---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MigrationMapping[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for resolveRecords. Apply mappings to bounded record groups, resolving semantic exceptions through evidence-backed decisions. Preserve identity and relationship provenance, isolate incompatible records, and avoid forcing all source variants into one target category for completion metrics. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
