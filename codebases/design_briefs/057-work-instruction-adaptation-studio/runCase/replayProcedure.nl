---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: InstructionVariant[] }
returns: StageResultProcedureScenario
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Generate ordinary and adverse scenarios involving local exceptions, missing resources, and interrupted work. Replay the adapted procedure to detect ambiguous instructions, skipped prerequisites, or branches that cannot reach a safe terminal state. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultProcedureScenario with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No scenario bypasses a retained safeguard or requires an unavailable capability.
