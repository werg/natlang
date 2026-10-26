---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: PracticeComplication[] }
returns: StageResultPracticeFeedback
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Compare the observable trajectory with procedure requirements and scenario facts. Explain specific effective and ineffective decisions, distinguish outcome luck from sound process, and propose a next scenario targeting demonstrated gaps rather than generic advice. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultPracticeFeedback with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Feedback cites actual actions and does not infer unobserved intentions.
