---
args: { goal: GoalContract, policy: PolicySelection, candidates: TriggerFinding[], evidence: EvidencePage[] }
returns: StageResultObligationTask
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Turn active obligations into bounded tasks with ownership, dependencies, and exact computed dates. Preserve legal interpretation separately from operational scheduling and identify impossible or conflicting task requirements before issuing commitments.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ObligationTask fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No unowned active obligation; dates use the configured calendar and explicit deadline rule.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
