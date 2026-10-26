---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RestorationOption[] }
returns: StageResultMonitoringTask
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Design monitoring that can distinguish progress, nonresponse, and unintended effects using supplied methods. Specify comparable locations, timing, and effort, and define what evidence would justify continuing, revising, or stopping an intervention. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMonitoringTask with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Monitoring can inform a real decision and does not confuse changing effort with changing conditions.
