---
args: { goal: GoalContract, policy: PolicySelection, candidates: PayableCase[], evidence: EvidencePage[] }
returns: StageResultInvoiceMatch
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Align descriptions, units, supplier references, and delivery context across invoice and order lines. Permit many-to-many matches for split shipments, but preserve ambiguous allocations and do not force every invoice line onto the nearest description.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual InvoiceMatch fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact unit conversions and sum conservation; candidate links retain provenance.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
