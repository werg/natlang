---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FabricationRoute[] }
returns: StageResultFabricationEstimate
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Use exact costing services with documented quantities, rates, waste assumptions, and supplier quotes. Distinguish fixed price from allowance or estimate, expose unknowns, and prevent a creative design change from silently invalidating the cost basis. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFabricationEstimate with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Quote arithmetic and scope match the proposed route and selected concept.
