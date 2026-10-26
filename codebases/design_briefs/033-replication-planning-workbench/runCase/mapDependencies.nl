---
args: { goal: GoalContract, policy: PolicySelection, candidates: MethodStep[], evidence: EvidencePage[] }
returns: StageResultReplicationDependency
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Identify data, software, configuration, and computational dependencies for each step. Detect unavailable versions and incompatible assumptions, then formulate approved alternatives without representing them as exact replication of the original environment.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ReplicationDependency fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Dependency graph is acyclic and all required resources are available or blocked.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
