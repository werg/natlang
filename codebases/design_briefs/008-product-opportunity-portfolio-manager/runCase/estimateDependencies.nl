---
args: { goal: GoalContract, policy: PolicySelection, candidates: Intervention[], evidence: EvidencePage[] }
returns: StageResultOpportunityEstimate
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Translate each intervention into bounded work packages, request estimates from authorized sources, and identify shared enabling work. Preserve cost ranges and unknowns rather than inventing precision; inspect whether dependencies make apparently independent projects coupled.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual OpportunityEstimate fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Shared work is costed once; missing estimates remain explicit and block firm commitments.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
