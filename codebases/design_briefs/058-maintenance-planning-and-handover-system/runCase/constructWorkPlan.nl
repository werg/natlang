---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MaintenanceProcedureFit[] }
returns: StageResultMaintenanceWorkOrder
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Decompose the approved procedure into accountable tasks, resource reservations, and verification steps. Sequence shared access and parts availability, preserve stop conditions, and distinguish planned work from authorized readiness to begin. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMaintenanceWorkOrder with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Exact resource feasibility and no bypass of required safety or qualification checks.
