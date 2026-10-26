---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for reconstructStock. Reconcile manifests, scans, picking notes, and receiving observations while preserving who observed what and when. Separate an item’s physical presence from its expected location and retain uncertainty when labels or serials are missing. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
