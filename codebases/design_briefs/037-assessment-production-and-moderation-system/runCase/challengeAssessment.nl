---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ScoringRubric[] }
returns: StageResultItemChallenge
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Independently solve each item, attempt reasonable alternate readings, and test rubric consistency on borderline responses. Repair items and rubrics together, then rerun coverage and duration checks before releasing a frozen assessment edition. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultItemChallenge with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No unresolved correctness or material ambiguity defect in released items.
