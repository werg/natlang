---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: PracticeTransition[] }
returns: StageResultPracticeComplication
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Select a complication appropriate to the learner’s progress and learning objective. Ensure it changes a meaningful decision, preserves a safe recovery path, and does not retroactively invalidate an action that was correct under earlier visible facts. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultPracticeComplication with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Complication is fair, procedurally relevant, and reproducible from the scenario seed.
