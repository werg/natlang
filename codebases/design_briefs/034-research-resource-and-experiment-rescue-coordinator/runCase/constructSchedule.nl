---
args: { goal: GoalContract, policy: PolicySelection, candidates: ResourceCapability[], evidence: EvidencePage[] }
returns: StageResultExperimentSlot
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Submit hard resource and timing constraints to an exact scheduler, then evaluate feasible schedules against supplied priorities and scientific dependencies. Keep optional preferences visible and explain unavoidable conflicts instead of silently dropping experiments.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExperimentSlot fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No double booking, expired sample window, or missing required operator.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
