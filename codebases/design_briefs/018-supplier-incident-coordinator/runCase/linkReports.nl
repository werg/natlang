---
args: { goal: GoalContract, policy: PolicySelection, candidates: SupplierIncident[], evidence: EvidencePage[] }
returns: StageResultIncidentCluster
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare complaints, inspections, and shipment events by product, lot, timing, and failure mechanism. Group plausibly related events while retaining independent incidents and contradictory evidence; do not merge all problems from one supplier into a single cause.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual IncidentCluster fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Cluster membership has a specific shared factor and preserves uncertain links.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
