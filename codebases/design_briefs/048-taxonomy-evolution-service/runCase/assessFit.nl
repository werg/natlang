---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultCategoryFit
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare incoming records with category definitions and examples, not names alone. Distinguish a genuinely new concept from a poorly worded existing category, and retain ambiguous membership where the taxonomy permits multiple dimensions. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultCategoryFit with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Classification uses the current definition version and preserves uncertainty.
