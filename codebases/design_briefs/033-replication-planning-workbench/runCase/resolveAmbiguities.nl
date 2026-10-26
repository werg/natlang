---
args: { goal: GoalContract, policy: PolicySelection, candidates: ReplicationDependency[], evidence: EvidencePage[] }
returns: StageResultMethodQuestion
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Enumerate plausible interpretations of missing details and assess whether each changes the result. Seek author clarification or documented evidence for consequential choices; retain sensitivity branches where ambiguity cannot be resolved.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual MethodQuestion fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No unacknowledged methodological invention or selective choice after seeing favorable results.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
