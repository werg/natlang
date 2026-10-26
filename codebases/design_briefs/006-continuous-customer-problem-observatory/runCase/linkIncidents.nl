---
args: { goal: GoalContract, policy: PolicySelection, candidates: ProblemEpisode[], evidence: EvidencePage[] }
returns: StageResultIncidentGroup
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare accounts for the same customer, event, and time window. Group follow-ups and copied messages into one incident while retaining repeat failures that actually occurred on different occasions; leave identity ambiguity unresolved.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual IncidentGroup fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Counts distinguish messages, incidents, and customers; no transitive merge across incompatible events.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
