---
args: { goal: GoalContract, policy: PolicySelection, candidates: CompatibilityFinding[], evidence: EvidencePage[] }
returns: StageResultQualificationTask
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Design the smallest supplier clarification, sample inspection, or approved test that resolves a decisive unknown. Specify pass criteria before results arrive and distinguish documentary clarification from testing requiring technical specialists.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual QualificationTask fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Test authority and acceptance criteria are explicit; no improvised hazardous test procedure.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
