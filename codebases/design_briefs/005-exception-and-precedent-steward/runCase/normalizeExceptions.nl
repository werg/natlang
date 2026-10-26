---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExceptionCase[], evidence: EvidencePage[] }
returns: StageResultExceptionRecord
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Extract the exact deviation, beneficiary, dates, and conditions of each granted exception. Keep the requirement that was waived separate from prerequisites that remained binding, and record whether the decision explicitly denied precedential value.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExceptionRecord fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: An exception never inherits broader scope than its authorization.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
