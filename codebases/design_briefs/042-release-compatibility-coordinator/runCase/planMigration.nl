---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CompatibilityTest[] }
returns: StageResultMigrationWave
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Construct staged migration plans with prerequisites, owners, compatibility shims where approved, and rollback conditions. Group consumers by meaningful risk and avoid advancing a wave before its required operational readiness evidence exists. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMigrationWave with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Each wave is feasible and has a tested rollback or explicit irreversible-change approval.
