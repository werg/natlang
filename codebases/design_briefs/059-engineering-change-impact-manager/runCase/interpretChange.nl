---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultDesignDelta
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare approved design revisions and change rationale to identify functional, interface, and process effects. Separate editorial drawing updates from meaningful changes and preserve requirements that remain binding despite a local modification. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDesignDelta with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every effect has a design-source basis and explicit scope.
