---
args: { goal: GoalContract, policy: PolicySelection, candidates: PayableDisposition[], evidence: EvidencePage[] }
returns: StageResultPostingReceipt
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare approved allocations with actual ledger and processor results. Reconcile lost acknowledgments by operation identity, detect subsequent credit notes, and maintain remaining obligations without retrying a payment whose outcome is unknown.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PostingReceipt fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No duplicate posting; ledger balance and approved allocation agree exactly.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
