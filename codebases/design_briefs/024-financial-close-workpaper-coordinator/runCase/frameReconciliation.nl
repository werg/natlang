---
args: { goal: GoalContract, policy: PolicySelection, candidates: CloseWorkpaper[], evidence: EvidencePage[] }
returns: StageResultReconciliationScope
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Interpret the purpose and supplied close policy for each workpaper. Identify the assertion to establish, relevant systems, exclusions, and escalation thresholds; keep documentary preparation separate from decisions reserved for qualified accounting reviewers.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ReconciliationScope fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Scope and source snapshots are explicit before analysis begins.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
