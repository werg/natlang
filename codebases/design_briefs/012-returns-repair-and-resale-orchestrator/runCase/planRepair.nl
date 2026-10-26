---
args: { goal: GoalContract, policy: PolicySelection, candidates: ReturnDisposition[], evidence: EvidencePage[] }
returns: StageResultRepairPlan
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Translate the selected path into authorized inspection or repair tasks using supplied procedures. Check part compatibility and workshop capacity, preserve stop conditions, and create alternative plans when repair would exceed delegated cost or capability.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual RepairPlan fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact parts and capacity feasibility; no improvised hazardous repair instruction.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
