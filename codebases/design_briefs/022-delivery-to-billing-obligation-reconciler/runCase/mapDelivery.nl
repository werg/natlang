---
args: { goal: GoalContract, policy: PolicySelection, candidates: BillingCondition[], evidence: EvidencePage[] }
returns: StageResultDeliveryEvidence
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Relate project artifacts, service logs, and customer messages to the promised deliverables. Identify partial or alternative delivery and distinguish a draft shared for comment from an accepted contractual handover.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual DeliveryEvidence fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Evidence supports the actual milestone, not merely related activity.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
