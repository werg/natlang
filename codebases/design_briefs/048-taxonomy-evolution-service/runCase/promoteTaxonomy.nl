---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: TaxonomyImpact[] }
returns: StageResultTaxonomyRelease
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Present impact, examples, and reporting consequences to the steward, then publish a versioned release and migration plan. Notify dependent systems, invalidate only relevant classification caches, and preserve old definitions for historical queries. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultTaxonomyRelease with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Approved definitions, migration scope, and dependent consumer compatibility are verified.
