---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FabricationConcept[] }
returns: StageResultFabricationRoute
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Translate each concept into a production route with material preparation, operations, finishing, assembly, and inspection. Identify outsourced steps and dependencies, reserve realistic capacity, and retain specialist approval requirements for consequential structural or safety questions. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFabricationRoute with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Exact resource and process feasibility; no unsupported engineering certification.
