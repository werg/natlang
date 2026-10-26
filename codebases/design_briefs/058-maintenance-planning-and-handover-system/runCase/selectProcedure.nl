---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AssetObservation[] }
returns: StageResultMaintenanceProcedureFit
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Match the asset variant and observed condition to approved procedures, checking revision, qualification, and required isolation or access conditions. Escalate unsupported cases rather than inventing maintenance instructions from general knowledge. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMaintenanceProcedureFit with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Correct asset and procedure edition with all critical prerequisites established.
