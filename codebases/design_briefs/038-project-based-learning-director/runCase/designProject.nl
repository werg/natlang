---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ProjectLearningGoal[] }
returns: StageResultProjectConcept
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Invent a coherent setting and challenge that naturally requires the target skills. Design meaningful roles and alternative solution paths, avoid making one learner a passive spectator, and include authentic constraints without introducing inaccessible prerequisites. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultProjectConcept with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: All roles contribute substantive evidence and the challenge is solvable with available resources.
