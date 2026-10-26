---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RepairSymptom[] }
returns: StageResultRepairDiagnostic
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Choose a bounded diagnostic task from supplied safe procedures that distinguishes plausible issues. Specify required qualifications, tools, and stop conditions; route work beyond cooperative capability to an appropriate specialist rather than improvising. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultRepairDiagnostic with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Diagnostic is approved, informative, and within assigned capability.
