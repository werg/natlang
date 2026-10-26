---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CategoryRevision[] }
returns: StageResultTaxonomyImpact
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reclassify only potentially affected records under baseline and candidate definitions, compare changes, and inspect boundary cases. Build explicit bridges for historical reporting and retain cases that cannot be mapped uniquely after a split. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultTaxonomyImpact with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Exact changed-record set and no fabricated historical precision.
