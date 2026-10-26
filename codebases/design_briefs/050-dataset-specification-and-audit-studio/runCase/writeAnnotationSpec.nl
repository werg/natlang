---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DatasetUse[] }
returns: StageResultAnnotationRule
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Draft operational labeling instructions with examples, counterexamples, and uncertainty handling. Separate observable labels from inferred attributes, define adjudication, and test whether independent annotators can apply the rule consistently. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultAnnotationRule with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Labels are grounded in available evidence and ambiguous cases have an explicit disposition.
