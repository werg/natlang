---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultProjectLearningGoal
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Interpret broad educational goals as observable project evidence and identify prerequisite skills. Separate the project’s fictional or practical theme from the competencies it must exercise, and preserve constraints on resources and learner workload. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultProjectLearningGoal with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every goal has a feasible demonstration within the project scope.
