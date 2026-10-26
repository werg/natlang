---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExperimentReview[], evidence: EvidencePage[] }
returns: StageResultExperimentDesign
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read the original plan and subsequent amendments in temporal order. Distinguish preregistered criteria from retrospective choices, identify the assignment and observation units, and preserve deviations that could change interpretation even when the experiment ran successfully.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExperimentDesign fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every design claim links to its effective plan version.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
