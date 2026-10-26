---
args: { goal: GoalContract, policy: PolicySelection, candidates: BillableAllocation[], evidence: EvidencePage[] }
returns: StageResultBillingPacket
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Assemble a customer-readable explanation and internal approval packet linking each amount to satisfied conditions. Recheck late amendments and acceptance changes before release, then reconcile the actual invoice identifier back to consumed obligations.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual BillingPacket fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Approved packet hash, current conditions, and exact ledger reconciliation required.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
