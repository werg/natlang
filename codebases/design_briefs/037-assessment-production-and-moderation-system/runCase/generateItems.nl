---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AssessmentBlueprint[] }
returns: StageResultAssessmentItem
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Create varied tasks that elicit the target evidence rather than superficial recall alone. Specify required context and resources, avoid hidden assumptions, and generate plausible alternative interpretations for subsequent review. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultAssessmentItem with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Item is answerable from supplied knowledge and permitted resources.
