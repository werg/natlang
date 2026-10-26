---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: PortfolioEvidence[] }
returns: StageResultCompetencyMapping
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare evidence with the actual competency definition and performance level. Recognize alternative demonstrations, avoid matching by keywords alone, and identify which facets are supported versus merely mentioned in a reflection. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultCompetencyMapping with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every mapping names the demonstrated behavior and source location.
