---
args: { goal: GoalContract, policy: PolicySelection, candidates: SubscriptionCase[], evidence: EvidencePage[] }
returns: StageResultChangeIntent
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read the request in conversational context and identify the intended account scope. Distinguish cancelling an add-on from closing the account, and ask a focused question only when ambiguity would change access, billing, or data retention.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ChangeIntent fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No retention offer or unrelated account action is inferred as consent.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
