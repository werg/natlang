---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: EnvironmentalAllocation[] }
returns: StageResultEnvironmentalClaim
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare proposed narrative claims with actual measured outcomes and activity evidence. Separate completion of an initiative from demonstrated environmental benefit, qualify estimated results, and reject persuasive claims unsupported by the reporting data. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultEnvironmentalClaim with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Each quantitative and impact claim is supported at the stated scope.
