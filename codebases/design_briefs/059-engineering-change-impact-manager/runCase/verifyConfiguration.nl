---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ChangeWorkPackage[] }
returns: StageResultChangeVerification
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare actual implemented configurations with approved applicability and acceptance criteria. Preserve authorized exceptions and unresolved field cases, then close the change only when required consumers and records reflect the correct revision state. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultChangeVerification with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No configuration claims based solely on document publication or task closure.
