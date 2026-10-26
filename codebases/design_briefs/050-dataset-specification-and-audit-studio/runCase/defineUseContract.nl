---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultDatasetUse
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Interpret the intended application and decisions the dataset will support. Define target population, task boundaries, prohibited uses, and meaningful evaluation slices; avoid treating easily available data as representative of the intended use. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDatasetUse with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every required slice and restriction has a use-based rationale.
