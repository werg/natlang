---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultSourceFieldMeaning
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile dictionaries, sample records, application notes, and domain-owner explanations. Identify fields whose meaning depends on record type or history, and distinguish null, not applicable, unknown, and deliberately withheld values before proposing transformations. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultSourceFieldMeaning with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No target mapping based solely on field names or observed value formats.
