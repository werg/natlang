---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: BehavioralChange[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for mapConsumers. Relate integrations, customer configurations, examples, and runbooks to the changed behavior. Identify implicit reliance such as ordering or retry assumptions, and preserve uncertain dependencies for targeted replay rather than declaring them unaffected. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
