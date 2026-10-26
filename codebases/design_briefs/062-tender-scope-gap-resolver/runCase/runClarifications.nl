---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ScopeInterface[] }
returns: StageResultScopeClarification
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Form precise clarification requests describing the disputed boundary and desired acceptance. Interpret replies at the commitment level, preserve qualifications, and avoid treating general cooperation language as acceptance of a specific responsibility. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultScopeClarification with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: A clarification closes a gap only with explicit scoped acceptance.
