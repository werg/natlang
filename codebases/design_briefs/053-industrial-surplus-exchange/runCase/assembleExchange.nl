---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReuseMatch[] }
returns: StageResultExchangePackage
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Construct feasible multi-party exchanges using exact quantity, reservation, and transport services. Account for dependent legs, storage windows, and adaptation costs, and preserve unallocated surplus and unmet needs explicitly. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultExchangePackage with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No double allocation, unsupported capacity, or negative quantity.
