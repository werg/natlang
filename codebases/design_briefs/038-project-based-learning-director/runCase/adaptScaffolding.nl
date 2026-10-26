---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ProjectMilestone[] }
returns: StageResultScaffoldingPlan
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Interpret progress artifacts to identify where support is needed without doing the substantive work for learners. Provide targeted examples, prompts, or task decomposition, and remove support when evidence shows independent capability. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultScaffoldingPlan with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Adaptation preserves assessment validity and does not substitute generated work for learner evidence.
