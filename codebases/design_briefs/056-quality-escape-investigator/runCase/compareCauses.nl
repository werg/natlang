---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: QualityExposure[] }
returns: StageResultQualityHypothesis
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Construct competing explanations from process history and defect patterns. Separate correlation with a shift, supplier, or operator from demonstrated cause, and formulate approved tests that could distinguish the mechanisms without unsafe experimentation. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultQualityHypothesis with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Causal claims remain provisional until supported by discriminating evidence.
