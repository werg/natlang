---
args: { goal: GoalContract, policy: PolicySelection, candidates: ConditionAssessment[], evidence: EvidencePage[] }
returns: StageResultReturnDisposition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Apply purchase terms and approved product-handling rules to condition evidence and customer preference. Identify cases needing specialist inspection, compare repair and replacement options, and avoid promising a remedy before required checks are satisfied.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ReturnDisposition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Route respects policy, customer consent, and approved handling restrictions.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
