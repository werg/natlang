---
args: { goal: GoalContract, policy: PolicySelection, candidates: IncidentCluster[], evidence: EvidencePage[] }
returns: StageResultExposureFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Trace potentially affected lots and orders through authoritative supply records. Distinguish confirmed exposure from precautionary scope, identify downstream commitments, and ask for missing traceability evidence before narrowing a containment boundary.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExposureFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact lineage traversal; no unsupported exclusion of potentially affected stock.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
