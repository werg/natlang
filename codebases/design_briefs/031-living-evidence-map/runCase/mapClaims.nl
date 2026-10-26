---
args: { goal: GoalContract, policy: PolicySelection, candidates: StudyObservation[], evidence: EvidencePage[] }
returns: StageResultEvidenceClaim
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Group evidence by the actual proposition and relevant population rather than broad topic. Identify incompatible measurements and contradictory findings, qualify applicability, and expose evidence gaps without treating absent studies as proof of no effect.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual EvidenceClaim fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Claims do not exceed the populations, designs, and measures of their supporting studies.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
