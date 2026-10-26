---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MaintenanceProgress[] }
returns: StageResultMaintenanceHandover
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Prepare a shift handover describing actual asset state, work in progress, unresolved findings, and required next checks. Require acknowledgment from the receiving owner and preserve the distinction between maintenance completion and authorized return to service. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMaintenanceHandover with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Handover includes all open obligations and current safety-relevant state.
