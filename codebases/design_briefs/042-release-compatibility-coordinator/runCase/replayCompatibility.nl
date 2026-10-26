---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ConsumerDependency[] }
returns: StageResultCompatibilityTest
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Execute representative consumer scenarios against baseline and candidate in isolated environments. Compare intended outcomes rather than raw output equality alone, and identify whether a difference is acceptable, breaking, or dependent on an unresolved contract. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultCompatibilityTest with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Paired replay uses identical inputs and independent consumer acceptance criteria.
