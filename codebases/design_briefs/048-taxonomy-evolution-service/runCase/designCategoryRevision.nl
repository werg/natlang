---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: TaxonomyDefect[] }
returns: StageResultCategoryRevision
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Draft revised definitions, inclusion and exclusion rules, and parent-child relationships. Preserve stable category IDs where meaning remains stable, create new identities when meaning changes, and explicitly map split or merged historical categories. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultCategoryRevision with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No cycle, contradictory boundary, or silent reuse of an ID for a different concept.
