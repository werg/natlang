---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultFacilityObservation
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Read occupant descriptions to identify location, symptom, timing, and practical impact. Distinguish observed behavior from guessed causes, resolve ambiguous room names, and preserve access or communication preferences without inferring sensitive personal circumstances. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFacilityObservation with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Location and symptom are established or a targeted clarification is requested.
