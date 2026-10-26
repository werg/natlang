---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AnalysisMeaning[] }
returns: StageResultDataSuitability
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Inspect schemas, dictionaries, representative records, and access constraints to determine whether the question is answerable. Distinguish absent data from inaccessible data and identify proxy fields that would change the meaning rather than silently substituting them. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDataSuitability with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Each required concept has a supported mapping or an explicit unanswerable component.
