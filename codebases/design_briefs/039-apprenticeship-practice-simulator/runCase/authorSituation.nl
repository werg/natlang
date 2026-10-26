---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultPracticeScenario
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Construct a realistic situation requiring the supplied procedure and meaningful information gathering. Separate facts visible at the start from facts revealed by appropriate questions, and include complications that test judgment without requiring unauthorized actions. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultPracticeScenario with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Scenario has a feasible compliant path and no hidden prerequisite the learner cannot discover.
