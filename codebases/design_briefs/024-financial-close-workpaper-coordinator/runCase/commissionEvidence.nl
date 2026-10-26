---
args: { goal: GoalContract, policy: PolicySelection, candidates: ReconciliationBreak[], evidence: EvidencePage[] }
returns: StageResultCloseTask
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Translate each consequential unresolved break into a bounded investigation with an accountable owner. Specify evidence capable of proving the explanation, prioritize dependencies on other close tasks, and expose unresolved issues before sign-off deadlines.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual CloseTask fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every material break has an accepted owner and completion criterion.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
