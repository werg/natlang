---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DefectObservation[] }
returns: StageResultQualityExposure
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Traverse exact lot, material, and process lineage to identify confirmed and precautionary exposure. Interpret process changes and rework histories, and avoid narrowing the scope merely because records are missing for some potentially affected units. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultQualityExposure with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Traceability is complete or the uncertainty is explicitly included in containment scope.
