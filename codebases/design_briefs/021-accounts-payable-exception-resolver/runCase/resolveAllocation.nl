---
args: { goal: GoalContract, policy: PolicySelection, candidates: PayableDiscrepancy[], evidence: EvidencePage[] }
returns: StageResultAllocationProposal
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Construct a partial resolution for portions supported by delivery and authorization evidence. Preserve disputed balances, prevent the same receipt from supporting multiple payments, and request targeted confirmation for unresolved acceptance or scope.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AllocationProposal fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Accepted plus held amounts equal the invoice balance after verified credits.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
