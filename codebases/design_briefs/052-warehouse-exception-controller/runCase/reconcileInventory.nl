---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: WarehouseWorkOrder[] }
returns: StageResultInventoryResolution
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare completed tasks with actual scans and custody events, apply approved exact adjustments, and verify conservation across locations. Preserve unresolved differences and distinguish a found item from a fulfilled customer order. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultInventoryResolution with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Physical movement and ledger changes reconcile without duplicate stock creation.
