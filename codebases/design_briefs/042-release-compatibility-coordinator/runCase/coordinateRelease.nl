---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MigrationWave[] }
returns: StageResultReleaseReadiness
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile code, documentation, support guidance, and customer notices into one release state. Explain audience-specific consequences and verify that approved migrations and communications match the exact release artifact being promoted. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultReleaseReadiness with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No release readiness claim over mismatched versions or unresolved blocking compatibility failures.
