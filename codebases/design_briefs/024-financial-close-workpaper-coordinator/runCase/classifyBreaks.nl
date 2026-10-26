---
args: { goal: GoalContract, policy: PolicySelection, candidates: ReconciliationScope[], evidence: EvidencePage[] }
returns: StageResultReconciliationBreak
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Consume exact reconciliation differences and inspect supporting narratives. Distinguish timing, mapping, missing postings, and unsupported explanations; group related breaks only when they share a demonstrated mechanism rather than the same amount.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ReconciliationBreak fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Numerical equality alone does not establish a valid match or explanation.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
