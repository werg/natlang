---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: TestFailureFinding[] }
returns: StageResultSuiteQualification
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Run independent seeded defects and mutation tests to evaluate whether the suite detects meaningful violations. Remove redundant tests only when coverage remains, preserve regression cases for repaired bugs, and report uncovered requirements explicitly. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultSuiteQualification with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Suite catches required seeded defects and does not pass merely because all tests match current behavior.
