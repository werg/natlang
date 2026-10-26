---
args: { goal: GoalContract, policy: PolicySelection, candidates: OutcomeDifference[], evidence: EvidencePage[] }
returns: StageResultBoundaryScenario
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Construct near-neighbor cases around changed thresholds, exceptions, and overlapping categories. Vary one consequential fact at a time, include adverse interactions, and identify scenarios whose outcome exposes an ambiguity requiring a policy decision.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual BoundaryScenario fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Perturbation is controlled and does not smuggle in multiple explanatory changes.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
