---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: PracticeScenario[] }
returns: StageResultLearnerAction
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Interpret the learner’s natural-language response against the available action set. Distinguish a proposed action from an executed one, ask for clarification when ambiguity affects consequences, and avoid rewarding jargon without the corresponding operational behavior. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultLearnerAction with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Action maps to a permitted typed operation or remains unexecuted.
