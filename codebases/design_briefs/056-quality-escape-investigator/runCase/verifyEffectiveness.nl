---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: QualityActionPlan[] }
returns: StageResultQualityClosure
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare implementation evidence and subsequent production observations with the predicted corrective effect. Detect recurrence and unintended consequences, retain unresolved exposure, and require qualified approval before lifting holds or closing the investigation. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultQualityClosure with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Closure requires verified effectiveness, not only completed action checkboxes.
