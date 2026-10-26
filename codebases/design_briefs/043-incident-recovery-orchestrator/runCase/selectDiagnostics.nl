---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: IncidentHypothesis[] }
returns: StageResultDiagnosticAction
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Choose from approved read-only or bounded diagnostic operations, considering expected discrimination and operational risk. Validate scope and rate limits, request commander approval for intrusive checks, and record what each possible result would change. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDiagnosticAction with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Diagnostic action is allowlisted and cannot widen credentials or query scope.
