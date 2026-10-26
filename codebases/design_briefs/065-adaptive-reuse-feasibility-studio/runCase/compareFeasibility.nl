---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReuseInvestigation[] }
returns: StageResultReuseOptionReview
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare concepts using exact supplied estimates, known constraints, functional benefits, and unresolved conditions. Present phased or contingent options and clear next decisions, separating creative promise from verified technical or regulatory feasibility. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultReuseOptionReview with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No unqualified feasibility or compliance claim beyond available professional evidence.
