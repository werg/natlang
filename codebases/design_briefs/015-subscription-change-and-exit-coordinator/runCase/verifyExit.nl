---
args: { goal: GoalContract, policy: PolicySelection, candidates: SubscriptionOperation[], evidence: EvidencePage[] }
returns: StageResultExitReceipt
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare actual subscription states and upcoming invoices with the selected option. Explain surviving obligations and access explicitly, provide a durable receipt, and reopen inconsistencies rather than declaring success after merely submitting a cancellation request.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExitReceipt fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: All affected subscriptions match the confirmed choice; no unapproved service remains active.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
