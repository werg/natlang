---
args: { goal: GoalContract, policy: PolicySelection, candidates: MeasurementFinding[], evidence: EvidencePage[] }
returns: StageResultResultClaim
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Consume approved statistical outputs rather than calculating inference through prose. Relate results to the specified estimand, inspect multiplicity and subgroup selection, and distinguish evidence of little effect from an underpowered or compromised experiment.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ResultClaim fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Claimed population, direction, and uncertainty match the actual computed analysis.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
