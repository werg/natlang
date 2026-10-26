---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultHandoverRequirement
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Read the supplied handover specification, asset schedules, and contractual requirements to identify required documents, tests, training, and approvals. Separate documentary requirements from technical sign-off and preserve conditions specific to asset classes or building areas. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultHandoverRequirement with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every required handover item has an explicit source and acceptance condition.
