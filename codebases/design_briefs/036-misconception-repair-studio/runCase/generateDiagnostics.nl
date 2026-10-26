---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MisconceptionHypothesis[] }
returns: StageResultDiagnosticTask
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Construct a short diagnostic task varying the relevant conceptual feature while controlling unrelated difficulty. Include an independently checked answer space and accessible wording; avoid hints that reveal which hypothesis the system expects to confirm. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDiagnosticTask with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Diagnostic outcomes discriminate hypotheses and do not introduce unlearned prerequisites.
