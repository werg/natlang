---
args: { goal: GoalContract, policy: PolicySelection, candidates: StudyIdentity[], evidence: EvidencePage[] }
returns: StageResultStudyObservation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Apply eligibility to available evidence, escalating insufficient abstracts to full-text review. Extract outcomes, methods, and limitations with source locations; keep author interpretation separate from directly reported observations and exact numerical results.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual StudyObservation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exclusions have criterion-specific reasons; extracted values preserve units and uncertainty.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
