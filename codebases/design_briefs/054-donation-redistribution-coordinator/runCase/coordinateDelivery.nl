---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DonationAllocation[] }
returns: StageResultDonationDelivery
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Arrange consented collection or delivery, verify availability before dispatch, and track actual recipient acceptance. Reallocate declined or unavailable items through the same policy rather than informal side agreements, and preserve privacy in donor communications. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDonationDelivery with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Delivery and acceptance are distinct; recipient details are shared only as needed.
