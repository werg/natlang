---
args: { goal: GoalContract, policy: PolicySelection, candidates: RequirementConflict[], evidence: EvidencePage[] }
returns: StageResultCompromiseOption
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Invent alternatives such as phased delivery, differentiated modes, or reduced scope while preserving mandatory constraints. Explicitly identify who gives up what and which new assumptions the compromise introduces; include a no-agreement option.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual CompromiseOption fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No concession is represented as accepted until the affected stakeholder confirms it.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
