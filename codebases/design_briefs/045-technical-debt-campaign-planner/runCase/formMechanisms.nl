---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DebtSymptom[] }
returns: StageResultDebtMechanism
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Group symptoms by plausible shared mechanisms such as unstable interfaces or manual coordination. Distinguish architectural causes from staffing or process issues, and formulate checks that could disconfirm the proposed technical explanation. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDebtMechanism with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: A popular explanation is not treated as established causality.
