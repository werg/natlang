---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: IncidentImpact[] }
returns: StageResultIncidentHypothesis
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Generate a bounded set of causal hypotheses from deployment history, dependencies, and symptoms. Track evidence for and against each, choose checks that distinguish mechanisms, and avoid repeatedly confirming the most recent or most fluent explanation. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultIncidentHypothesis with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Hypotheses remain provisional until diagnostic evidence supports them.
