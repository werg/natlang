---
args: { goal: GoalContract, policy: PolicySelection, candidates: EligibilityCriterion[], evidence: EvidencePage[] }
returns: StageResultStudyIdentity
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare authors, enrollment windows, populations, identifiers, and methods to determine whether reports describe the same study or overlapping cohorts. Preserve uncertain links and distinguish follow-up analyses from independent replications before counting evidence.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual StudyIdentity fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Multiple reports cannot inflate the count of independent studies.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
