---
args: { goal: GoalContract, policy: PolicySelection, candidates: CorrectiveAction[], evidence: EvidencePage[] }
returns: StageResultIncidentClosure
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Check implemented actions, release criteria, and subsequent observations. Preserve residual risk and monitoring obligations; reopen the incident when a materially similar recurrence contradicts the closure rationale rather than treating it as a fresh unrelated ticket.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual IncidentClosure fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: All containment and verification obligations resolved or explicitly transferred to an accepted owner.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
