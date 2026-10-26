---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DiagnosticTask[] }
returns: StageResultLearningIntervention
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Select a suitable explanation and contrastive examples, then create practice with gradually reduced scaffolding. Adapt to observed responses without simply repeating the original answer, and escalate persistent ambiguity to the instructor. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultLearningIntervention with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Explanations and examples are correct under an independent subject-matter check.
