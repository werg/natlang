---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ShippingAlternative[] }
returns: StageResultShippingAllocation
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Submit candidate plans to exact capacity and timing services, then compare feasible allocations under supplied priorities. Reserve selected capacity before promising it, retain unfulfilled demand explicitly, and detect conflicts across shipments sharing scarce resources. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultShippingAllocation with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Capacity conservation, route feasibility, and exact cost limits pass.
