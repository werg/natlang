---
args: { goal: GoalContract, policy: PolicySelection, candidates: AssumptionBinding[], evidence: EvidencePage[] }
returns: StageResultChangeImpact
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare old and new assertions under their dates, scope, and authority. Determine which premise is actually invalidated and retain the part of the decision whose conditions still hold; never treat a synonym edit as a substantive reversal.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ChangeImpact fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact interval overlap plus semantic premise contradiction, not keyword overlap alone.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
