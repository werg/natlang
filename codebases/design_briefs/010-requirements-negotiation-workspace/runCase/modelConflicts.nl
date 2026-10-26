---
args: { goal: GoalContract, policy: PolicySelection, candidates: StakeholderPosition[], evidence: EvidencePage[] }
returns: StageResultRequirementConflict
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Identify requirements that cannot jointly hold under the supplied technical, time, or resource constraints. Distinguish true contradictions from terminology differences and propose the smallest clarification or relaxation that would make the pair compatible.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual RequirementConflict fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact conflicts have a constraint witness; semantic conflicts retain explanatory counterexamples.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
