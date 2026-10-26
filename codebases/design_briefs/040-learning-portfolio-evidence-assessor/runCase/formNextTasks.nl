---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CompetencyFinding[] }
returns: StageResultEvidenceTask
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Design the smallest authentic task that fills a consequential evidence gap. Tailor context and resources to the learner while preserving the competency, and specify what would count as success before the new work is submitted. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultEvidenceTask with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Task directly addresses the missing facet and is feasible under stated constraints.
