---
args: { goal: GoalContract, policy: PolicySelection, candidates: RecoveryCase[], evidence: EvidencePage[] }
returns: StageResultServiceIssue
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Reconstruct the timeline across messages and split failures only when they can be resolved independently. Retain earlier promises and interactions between issues, such as a late replacement making a promotional remedy no longer useful.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ServiceIssue fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every issue links to the relevant order and the customer’s stated outcome.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
