---
args: { goal: GoalContract, policy: PolicySelection, candidates: MethodQuestion[], evidence: EvidencePage[] }
returns: StageResultReplicationTask
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Translate the reviewed procedure into typed tasks using approved computational capabilities. Pin inputs and environments, define checkpoint and output contracts, and include resource limits and recovery behavior for failed jobs.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ReplicationTask fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Commands execute only in the sandbox and consume the pinned dependency manifest.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
