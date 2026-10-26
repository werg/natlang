---
args: { goal: GoalContract, policy: PolicySelection, candidates: ReplicationTask[], evidence: EvidencePage[] }
returns: StageResultReplicationFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare outputs with the original reported quantities and tolerances, distinguishing implementation failures from substantive differences. Investigate divergences through minimal controlled changes and publish uncertainty rather than claiming replication from superficial similarity.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ReplicationFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact comparisons and methodological provenance support every reported match or discrepancy.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
