---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AnnotationRule[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for auditCandidates. Inspect candidate records for task relevance, data quality, provenance, consent, and leakage risks. Distinguish missing coverage from invalid records, and quarantine material whose rights or source integrity cannot be established. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
