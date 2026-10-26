---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: LearnerAction[] }
returns: StageResultPracticeTransition
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Apply the exact simulator’s legal transition rules, then formulate context-appropriate responses from actors who know only their own information. Preserve consequences across turns and do not reveal hidden facts merely to help the learner succeed. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultPracticeTransition with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Resource, knowledge, and procedure invariants remain valid after every transition.
