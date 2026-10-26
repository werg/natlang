---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DebtWave[] }
returns: StageResultDebtOutcome
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare post-change workflow evidence with the original problem definition, accounting for changes in workload and observation. Distinguish successful delivery from actual improvement and reopen mechanisms that the completed work did not address. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDebtOutcome with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Completion requires evidence of the intended operational effect, not only task closure.
