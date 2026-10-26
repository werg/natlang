---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FacilityVisit[] }
returns: StageResultFacilityResolution
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare technician completion evidence and subsequent occupant observations with the original complaint. Distinguish a replaced component from resolution of the symptom, detect recurrence, and reopen only the relevant issue when work does not achieve the intended outcome. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFacilityResolution with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Closure requires the defined outcome evidence or an explicit accepted monitoring state.
