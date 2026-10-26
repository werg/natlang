---
args: { goal: GoalContract, policy: PolicySelection, candidates: InvoiceMatch[], evidence: EvidencePage[] }
returns: StageResultPayableDiscrepancy
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Interpret differences using contract terms and receiving evidence. Distinguish partial delivery, unauthorized substitution, freight charges, and genuine duplication; a repeated invoice number is a signal to investigate rather than a sufficient payment decision.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PayableDiscrepancy fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every disputed amount derives from exact line calculations and a named discrepancy.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
