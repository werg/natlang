---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RestorationObjective[] }
returns: StageResultRestorationOption
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Propose staged interventions from approved guidance and known site conditions. Explain the hypothesized mechanism, dependencies, resource needs, and stop conditions; route unfamiliar or potentially harmful interventions to qualified specialists instead of improvising. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultRestorationOption with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Known ecological and access constraints respected; no unsupported intervention prescription.
