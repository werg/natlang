---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultDonationOffer
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Interpret offers and condition evidence, distinguishing available items from tentative promises. Record quantities, collection constraints, and any permitted-use restrictions; request missing information that changes suitability rather than accepting vague offers into inventory. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDonationOffer with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Only confirmed quantities enter allocatable stock.
