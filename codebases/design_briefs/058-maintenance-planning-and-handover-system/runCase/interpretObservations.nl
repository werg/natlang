---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultAssetObservation
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile inspection notes, service history, and operator reports to identify observed conditions and unresolved discrepancies. Separate symptoms from diagnosis, retain intermittent behavior, and identify information needed for routing under approved maintenance procedures. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultAssetObservation with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Observations do not become established faults without supporting evidence.
