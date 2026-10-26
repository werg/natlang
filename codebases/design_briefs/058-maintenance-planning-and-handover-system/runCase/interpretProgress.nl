---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MaintenanceWorkOrder[] }
returns: StageResultMaintenanceProgress
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare technician notes and system events with task completion criteria. Distinguish work attempted, work completed, and verification passed; preserve changed conditions and unexpected findings that alter the remaining plan. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMaintenanceProgress with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No completion based solely on free-text optimism or elapsed time.
