---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AssessmentItem[] }
returns: StageResultScoringRubric
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Define observable scoring criteria and partial-credit boundaries, separating conceptual correctness from presentation preferences. Include correct alternative methods and common errors, and state how ambiguity or missing evidence should be handled. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultScoringRubric with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Rubric does not penalize a valid solution merely for differing from the model answer.
