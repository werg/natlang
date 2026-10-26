---
args: { goal: GoalContract, policy: PolicySelection, candidates: SupplierOffer[], evidence: EvidencePage[] }
returns: StageResultSourcingPackage
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Use exact allocation services to enumerate feasible combinations, then compare operational resilience and semantic fit under supplied priorities. Consider split supply and alternative delivery patterns without creating allocations that suppliers have not offered capacity to fulfill.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual SourcingPackage fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Quantity conservation, capacity limits, and conditional-offer compatibility all pass.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
