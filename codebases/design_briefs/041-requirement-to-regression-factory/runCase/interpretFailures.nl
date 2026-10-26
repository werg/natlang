---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RegressionTest[] }
returns: StageResultTestFailureFinding
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare failures with requirements, execution traces, and fixture health. Distinguish a product defect from a bad oracle, flaky setup, or unavailable dependency; propose the smallest repair without weakening a valid assertion to match defective behavior. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultTestFailureFinding with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Failure classification is evidence-backed and environment errors are not counted as product coverage.
