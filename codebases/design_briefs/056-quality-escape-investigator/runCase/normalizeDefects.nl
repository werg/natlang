---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultDefectObservation
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile customer reports, operator notes, and inspection descriptions into comparable observed defects. Preserve measurement conditions and terminology differences, distinguish symptoms from suspected causes, and retain reports that may describe a different mechanism. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDefectObservation with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every defect observation remains tied to its original inspection context.
