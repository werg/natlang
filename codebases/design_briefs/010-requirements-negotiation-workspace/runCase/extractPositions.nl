---
args: { goal: GoalContract, policy: PolicySelection, candidates: RequirementsProject[], evidence: EvidencePage[] }
returns: StageResultStakeholderPosition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read each request with its rationale and authority. Separate desired outcomes, proposed solutions, non-negotiable constraints, and preferences; do not upgrade emphatic language into a mandatory requirement without an explicit basis.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual StakeholderPosition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Each strength classification has evidence or remains unknown.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
