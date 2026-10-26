---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DebtWorkPackage[] }
returns: StageResultDebtWave
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Order enabling changes, migrations, and cleanup using exact dependency and capacity services. Preserve ongoing delivery constraints, identify irreversible steps, and create milestones that deliver useful improvements before the entire campaign is complete. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDebtWave with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Feasible capacity, acyclic dependencies, and safe migration sequencing.
