---
args: { goal: GoalContract, policy: PolicySelection, candidates: ClausePackage[], evidence: EvidencePage[] }
returns: StageResultRedlineFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Interpret counterparty edits in context, separating stylistic changes from altered obligations or risk allocation. Recompute package interactions, identify conditional acceptance, and route edits beyond the mandate to qualified reviewers.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual RedlineFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No unreviewed material change hidden as editorial cleanup.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
