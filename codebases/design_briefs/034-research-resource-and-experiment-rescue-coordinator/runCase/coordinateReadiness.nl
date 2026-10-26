---
args: { goal: GoalContract, policy: PolicySelection, candidates: RescueOption[], evidence: EvidencePage[] }
returns: StageResultReadinessCheck
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Collect readiness confirmations from resource owners and investigators, verify current status immediately before execution, and issue role-specific instructions. Preserve cancellations and rescheduling history so downstream analyses know which actual conditions applied.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ReadinessCheck fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: A scheduled slot is not a go decision; all required readiness evidence must be current.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
