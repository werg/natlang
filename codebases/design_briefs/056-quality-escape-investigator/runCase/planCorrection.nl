---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: QualityHypothesis[] }
returns: StageResultQualityActionPlan
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Select approved containment and corrective actions, assign qualified owners, and specify verification criteria before implementation. Distinguish immediate correction of defective units from prevention of recurrence and preserve unaffected production where evidence supports it. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultQualityActionPlan with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Actions respect authority, safety prerequisites, and explicit release conditions.
