---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultPortfolioEvidence
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Read artifacts, reflections, and supervisor comments to identify observable work and its context. Separate the learner’s own contribution from group output, distinguish direct evidence from self-report, and preserve uncertainty about authorship or assistance. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultPortfolioEvidence with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Evidence type and authorship status are explicit.
