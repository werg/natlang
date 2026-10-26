---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: EnvironmentalMeasure[] }
returns: StageResultEnvironmentalAllocation
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Allocate shared activities using supplied rules and exact calculations, checking for duplicate records or overlapping contractor reports. Distinguish a transferred responsibility from a duplicated physical activity and preserve unresolved attribution rather than balancing by assumption. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultEnvironmentalAllocation with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Exact totals and no double-counting across reporting boundaries.
