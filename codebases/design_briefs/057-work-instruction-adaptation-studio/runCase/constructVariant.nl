---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: LocalProcedureConstraint[] }
returns: StageResultInstructionVariant
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Rewrite steps for the local context using approved terminology and resources. Preserve sequence, guards, stop conditions, and verification; represent genuine deviations explicitly rather than hiding them in friendlier wording or inferred substitutions. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultInstructionVariant with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: All master safety and prerequisite obligations survive unless explicitly superseded by authorized engineering change.
