---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ShippingAllocation[] }
returns: StageResultShippingRecovery
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Sequence approved bookings and communications, reconcile carrier acknowledgments, and track actual shipment progress against revised promises. Replan only affected legs when new events arrive and explain residual uncertainty rather than claiming recovery after booking alone. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultShippingRecovery with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Actual carrier state and customer-approved commitments agree before closure.
