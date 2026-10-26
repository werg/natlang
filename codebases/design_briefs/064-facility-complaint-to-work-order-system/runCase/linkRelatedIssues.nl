---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FacilityObservation[] }
returns: StageResultFacilityIssueGroup
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare reports by location, asset dependencies, timing, and symptom mechanism. Group likely shared issues for coordinated investigation while retaining unrelated complaints that use similar language; do not close a new report solely because a nearby task was completed. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFacilityIssueGroup with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Grouping has evidence and preserves independent outcomes.
