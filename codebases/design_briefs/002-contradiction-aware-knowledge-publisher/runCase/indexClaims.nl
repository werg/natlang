---
args: { goal: GoalContract, policy: PolicySelection, candidates: KnowledgeRevision[], evidence: EvidencePage[] }
returns: StageResultScopedClaim
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Extract independently checkable claims with qualifications and page context. Resolve entity references conservatively, attach author and effective period, and keep instructions separate from descriptions of historical behavior or quoted third-party views.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ScopedClaim fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Claims retain every material negation, exception, and audience qualifier.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
