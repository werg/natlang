---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: StockIdentityFinding[] }
returns: StageResultWarehouseFinding
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Interpret the discrepancy against confirmed identity and movement evidence. Separate damage from quantity shortage and pending receipt from true absence; identify which inventory and fulfillment commitments may be affected without broad unsupported holds. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultWarehouseFinding with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Exact affected quantities and evidence-backed exception categories.
