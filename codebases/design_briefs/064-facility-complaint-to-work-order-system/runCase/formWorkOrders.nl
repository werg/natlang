---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FacilityIssueGroup[] }
returns: StageResultFacilityWorkOrder
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Choose approved inspection or repair task types and required trade qualifications. Sequence dependencies, include access arrangements and occupant impact, and route safety-sensitive or uncertain cases to qualified personnel rather than generating improvised repairs. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFacilityWorkOrder with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Task scope and qualification match the observed issue and approved procedure.
