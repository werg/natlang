---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ObservationLink[] }
returns: StageResultObservationQuestion
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Form targeted follow-up requests that could distinguish plausible interpretations or correct metadata. Use respectful wording, avoid leading the observer toward the expected label, and route specialist-sensitive cases for expert review. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultObservationQuestion with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Questions are informative, feasible, and do not request unsafe field activity.
