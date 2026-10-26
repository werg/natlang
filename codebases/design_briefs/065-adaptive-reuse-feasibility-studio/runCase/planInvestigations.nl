---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReuseConcept[] }
returns: StageResultReuseInvestigation
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Identify the surveys, specialist reviews, or evidence needed to distinguish feasible concepts. Prioritize investigations by their ability to change a decision, and specify how positive, negative, or inconclusive findings would alter the concept. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultReuseInvestigation with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Each investigation resolves a named uncertainty and uses qualified professionals where required.
