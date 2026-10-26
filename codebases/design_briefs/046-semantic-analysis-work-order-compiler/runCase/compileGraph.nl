---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DataSuitability[] }
returns: StageResultAnalysisWorkGraph
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Construct a typed DAG of exact filters, joins, bounded semantic classification, temporal matching, and aggregation. Push exact restrictive filters before semantic work when meaning is preserved; attach provenance and missing-data rules to every edge and reject unsupported operations. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultAnalysisWorkGraph with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Graph typechecks, has no cycle, and preserves the formalized population and output grain.
