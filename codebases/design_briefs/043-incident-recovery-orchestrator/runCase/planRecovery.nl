---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DiagnosticAction[] }
returns: StageResultRecoveryStep
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Select a recovery path from approved runbooks using current evidence and legal action options. Sequence reversible actions first where appropriate, preserve prerequisites, and avoid improvising destructive commands or treating a successful command as recovered service. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultRecoveryStep with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Current authority, service state, and runbook preconditions checked immediately before execution.
