---
args: { goal: GoalContract, policy: PolicySelection, candidates: OpportunityPortfolio[], evidence: EvidencePage[] }
returns: StageResultCustomerNeed
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Reconcile customer requests into underlying outcomes and constraints. Separate a suggested implementation from the need it serves; preserve segments whose superficially similar requests arise from different workflows and require different interventions.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual CustomerNeed fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every need has attributable evidence; invented segments and unsupported market sizes are forbidden.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
