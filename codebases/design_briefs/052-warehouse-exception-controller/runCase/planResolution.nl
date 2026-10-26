---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: WarehouseFinding[] }
returns: StageResultWarehouseWorkOrder
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Create bounded search, recount, relabel, inspection, or transfer tasks using approved warehouse procedures. Sequence dependencies and require independent confirmation for consequential inventory adjustments or uncertain identity changes. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultWarehouseWorkOrder with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No inventory adjustment merely to make a reconciliation balance.
