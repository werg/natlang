---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ProcedureRequirement[] }
returns: StageResultLocalProcedureConstraint
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare local equipment, materials, staffing, and terminology with the master procedure. Separate harmless naming differences from capability changes and identify where local conditions make a step impossible or require engineering review. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultLocalProcedureConstraint with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Local compatibility is verified; unknown critical differences block adaptation.
