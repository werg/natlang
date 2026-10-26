---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: StockObservation[] }
returns: StageResultStockIdentityFinding
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare descriptions, packaging, serials, lot information, and movement history to identify candidate stock. Distinguish a mislabeled item from a misplaced one, and request a discriminating scan or inspection rather than choosing the closest description. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultStockIdentityFinding with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Identity requires compatible evidence; near-matches remain unresolved.
