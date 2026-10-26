---
args: { goal: GoalContract, policy: PolicySelection, candidates: WorkpaperConclusion[], evidence: EvidencePage[] }
returns: StageResultWorkpaperSignoff
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Bind sign-off to exact source and conclusion versions. When a late journal or source correction arrives, trace affected assertions and invalidate only dependent approvals; preserve the prior signed workpaper as historical evidence.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual WorkpaperSignoff fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No current sign-off over stale inputs; reviewer independence enforced by the host.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
