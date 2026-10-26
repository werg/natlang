---
args: { goal: GoalContract, policy: PolicySelection, candidates: ProblemConcept[], evidence: EvidencePage[] }
returns: StageResultTrendSlice
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Define an observable population and comparable time windows before invoking exact aggregation. Interpret changes in collection channels and exposure; distinguish a rise in complaints from a rise in the complaint rate among eligible customers.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual TrendSlice fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Numerators are subsets of denominators; missing exposure forbids unsupported rate claims.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
