---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FacilityWorkOrder[] }
returns: StageResultFacilityVisit
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Schedule visits using exact availability and resource constraints, obtain required access consent, and coordinate multiple trades. Preserve cancellations and failed-access events, and update occupants with truthful windows rather than unsupported precise arrival promises. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFacilityVisit with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No unauthorized access or impossible trade sequence.
