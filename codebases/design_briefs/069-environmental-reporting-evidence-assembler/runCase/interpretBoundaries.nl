---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultReportingBoundary
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Read the supplied reporting framework, organizational boundary, and amendments to determine included activities and required measures. Distinguish ownership, operational control, and other supplied boundary concepts rather than choosing whichever yields a favorable result. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultReportingBoundary with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Boundary interpretation is explicit and approved before aggregation.
