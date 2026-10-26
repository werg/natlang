---
args: { goal: GoalContract, policy: PolicySelection, candidates: DecisionAssertion[], evidence: EvidencePage[] }
returns: StageResultAssumptionBinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: For each decision ask what would have to change to make reconsideration necessary. Separate essential premises from background rationale; emit a directed dependency with a counterfactual explanation and a passage establishing the connection.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AssumptionBinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every edge has source support; inferred edges remain proposals until owner review.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
