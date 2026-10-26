---
args: { goal: GoalContract, policy: PolicySelection, candidates: RecoveryAction[], evidence: EvidencePage[] }
returns: StageResultRecoveryClosure
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare actual outcomes with every agreed remedy and earlier promise. A refund request or shipment label is not completion; retain unresolved components and explain partial closure without resetting the entire case history.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual RecoveryClosure fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Close only when all mandatory outcomes are evidenced or an explicit alternative is accepted.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
