---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: TestPartition[] }
returns: StageResultRegressionTest
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Create executable test specifications using approved harness capabilities and isolated fixtures. Define deterministic setup, observable assertions, and cleanup; avoid brittle checks on incidental wording or timing unless those are actually contractual. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultRegressionTest with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Tests compile and have independent expected behavior, not output copied from the implementation.
