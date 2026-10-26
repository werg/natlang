---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FieldObservation[] }
returns: StageResultObservationInterpretation
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Interpret local names and descriptive phrases against project terminology and context. Retain multiple plausible identities when evidence is insufficient, and avoid forcing unusual observations into familiar categories merely to complete the dataset. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultObservationInterpretation with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Concept assignments have discriminating evidence or remain provisional.
