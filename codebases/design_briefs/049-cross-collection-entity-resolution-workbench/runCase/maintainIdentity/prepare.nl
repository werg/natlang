---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CanonicalEntity[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for maintainIdentity. Process new evidence and reviewer corrections as reversible identity changes. Trace downstream uses, prepare split or merge migrations, and preserve old identifiers as redirects with explicit validity rather than silently changing historical joins. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
