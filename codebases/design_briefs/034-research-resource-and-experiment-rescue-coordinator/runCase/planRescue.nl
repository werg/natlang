---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExperimentSlot[], evidence: EvidencePage[] }
returns: StageResultRescueOption
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: When a resource or sample becomes unavailable, generate alternatives such as resequencing, qualified substitution, or postponement. Explicitly state scientific deviations and seek investigator approval where the original protocol would no longer hold.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual RescueOption fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Recovery preserves essential conditions or records an approved protocol deviation.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
