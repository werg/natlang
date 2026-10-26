---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: LearningIntervention[] }
returns: StageResultTransferFinding
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Assess performance on new tasks that change surface details and require the intended concept. Distinguish independent transfer from success under hints, update the learner model conservatively, and retain evidence for instructor review. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultTransferFinding with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Mastery requires novel unassisted evidence under the supplied competency standard.
