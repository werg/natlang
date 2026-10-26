---
args: { goal: GoalContract, policy: PolicySelection, candidates: NegativeResultCase[], evidence: EvidencePage[] }
returns: StageResultResearchAttempt
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read plans, execution logs, and researcher notes to reconstruct the attempted test. Separate intended from actual procedure and identify incomplete steps, measurement failures, or deviations that limit what the result can establish.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ResearchAttempt fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: The archive records what actually happened, not only what was planned.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
