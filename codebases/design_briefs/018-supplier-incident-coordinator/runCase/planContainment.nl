---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExposureFinding[], evidence: EvidencePage[] }
returns: StageResultContainmentPlan
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Select approved containment actions proportionate to the established issue and uncertainty. Coordinate holds, inspections, alternative supply, and reviewed communications while preserving unaffected orders and avoiding promises before capacity is reserved.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ContainmentPlan fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Actions remain within delegated authority and have explicit release criteria.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
