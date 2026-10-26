---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ScopeDifference[] }
returns: StageResultConstructionImpact
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Trace design, procurement, trade interfaces, and schedule dependencies using exact project graphs. Judge which dependencies are materially affected, distinguish potential from confirmed delay, and identify evidence needed to support commercial or timing claims. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultConstructionImpact with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Claimed downstream effects have dependency paths and explicit assumptions.
