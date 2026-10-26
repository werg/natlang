---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ObservationInterpretation[] }
returns: StageResultObservationLink
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare observer, time, location, media hashes, and narrative context to identify repeated submissions of one event. Preserve independent observations of the same phenomenon and avoid merging nearby events solely because they share a likely identity. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultObservationLink with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Event counts distinguish duplicate submissions from independent corroboration.
