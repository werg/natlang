---
args: { goal: GoalContract, policy: PolicySelection, candidates: AllocationProposal[], evidence: EvidencePage[] }
returns: StageResultPayableDisposition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Prepare an auditable recommendation to pay, hold, request correction, or apply a verified credit. Route by entity and delegated thresholds, and explain why undisputed portions may proceed without treating the whole invoice as resolved.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PayableDisposition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Segregation of duties and current approvals enforced outside the semantic executor.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
