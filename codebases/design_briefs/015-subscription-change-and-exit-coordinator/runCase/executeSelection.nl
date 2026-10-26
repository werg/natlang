---
args: { goal: GoalContract, policy: PolicySelection, candidates: ChangeOption[], evidence: EvidencePage[] }
returns: StageResultSubscriptionOperation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Verify consent to the exact option and effective date, then apply changes through idempotent adapters. Recheck account versions before commitment and reconcile partial changes across linked products using compensating actions where permitted.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual SubscriptionOperation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No changed selection after consent; unknown outcomes block blind repetition.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
