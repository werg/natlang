---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ExchangePackage[] }
returns: StageResultReuseReceipt
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Coordinate approved transfers and verify quantity, condition, ownership, and recipient acceptance at handoff. Record rejected or partially accepted material and update future matching evidence without falsely reporting a completed reuse benefit. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultReuseReceipt with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Transfer and acceptance evidence support reported quantities and outcomes.
