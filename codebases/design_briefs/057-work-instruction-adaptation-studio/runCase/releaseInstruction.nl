---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ProcedureScenario[] }
returns: StageResultInstructionRelease
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Obtain qualified review of the exact variant, publish controlled copies, and coordinate acknowledgment or training. Retire outdated copies without erasing historical execution records and re-evaluate affected instructions when equipment or master procedures change. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultInstructionRelease with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Approved version and site applicability are visible at every point of use.
