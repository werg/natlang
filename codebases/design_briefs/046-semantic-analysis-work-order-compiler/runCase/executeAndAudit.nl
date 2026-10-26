---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AnalysisWorkGraph[] }
returns: StageResultAnalysisSlice
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Execute finite partitions, cache semantic judgments by record and rubric version, and reconcile every input disposition. Audit sampled positive and negative labels, exact join cardinalities, temporal boundaries, and denominators before materializing the final aggregate. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultAnalysisSlice with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No silent row loss, duplicate join inflation, unclassified inclusion, or unsupported denominator.
