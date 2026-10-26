---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MigrationDecision[] }
returns: StageResultMigrationRehearsal
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Materialize a staged target snapshot, validate referential integrity and business invariants, and compare representative source-target round trips. Rehearse rollback and delta capture so cutover does not lose changes made after the initial extraction. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMigrationRehearsal with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Exact counts, relationships, conservation constraints, and reversible mappings verified.
