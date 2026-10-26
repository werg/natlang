---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MigrationRehearsal[] }
returns: StageResultMigrationCutover
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Freeze the approved mapping and cutover scope, apply final source deltas, and reconcile target results before switching consumers. Track quarantined records and post-cutover corrections explicitly, retaining a safe rollback or forward-repair plan for irreversible changes. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMigrationCutover with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No unaccounted source record or unapproved mapping change during cutover.
