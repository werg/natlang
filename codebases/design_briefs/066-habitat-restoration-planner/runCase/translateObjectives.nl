---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HabitatObservation[] }
returns: StageResultRestorationObjective
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Convert landholder and ecological goals into observable objectives under supplied guidance. Separate desired long-term outcomes from near-term actions, identify conflicting uses, and preserve uncertainty about causal pathways rather than promising guaranteed ecological results. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultRestorationObjective with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Objectives have measurable indicators and accountable expert interpretation.
