---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultFieldObservation
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Parse field notes and structured submissions while preserving original wording, location precision, and observation method. Distinguish what was directly seen from inferred identity or cause, and flag missing metadata without inventing exact coordinates or times. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultFieldObservation with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Original evidence and uncertainty remain recoverable.
