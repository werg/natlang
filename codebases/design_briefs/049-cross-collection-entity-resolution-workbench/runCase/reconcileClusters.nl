---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: IdentityJudgment[] }
returns: StageResultIdentityCluster
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Build clusters using constrained graph operations rather than unconditional transitive closure. Before merging components, test all relevant cannot-link and temporal constraints; route inconsistent triangles to review and preserve uncertain alternatives. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultIdentityCluster with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No cluster contains an established cannot-link pair or impossible identity chronology.
