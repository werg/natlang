---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AnalysisSlice[] }
returns: StageResultAnalysisAnswer
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Formulate the result using exact computed values and the approved question interpretation. Explain exclusions, uncertainty, and unanswerable components; attach the executable work order and lineage so another analyst can reproduce or revise the analysis. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultAnalysisAnswer with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every numerical claim matches the materialized result and every semantic qualification matches the plan.
