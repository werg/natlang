---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: BehaviorObservation[] }
returns: StageResultDocumentationMismatch
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile requirements, release decisions, tests, and observed behavior to determine which artifact is wrong. Do not automatically rewrite documentation to match a bug; preserve unresolved intent questions for the responsible product or engineering owner. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDocumentationMismatch with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Repair direction has authoritative support rather than convenience.
