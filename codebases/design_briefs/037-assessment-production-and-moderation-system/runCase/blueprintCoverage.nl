---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultAssessmentBlueprint
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Translate the assessment purpose into observable competencies and a balanced task blueprint. Distinguish intended skills from incidental reading or tool demands, and identify constraints on duration, accessibility, and allowable assistance. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultAssessmentBlueprint with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every weighted competency has a measurable evidence path.
