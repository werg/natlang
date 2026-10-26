---
args: { goal: GoalContract, policy: PolicySelection, candidates: MeasureDefinition[], evidence: EvidencePage[] }
returns: StageResultMeasureRelation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare constructs, scales, recall periods, and populations against the intended analysis. Separate exact unit conversion from inferential harmonization, and identify subsets where comparability holds without claiming it for the entire dataset.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual MeasureRelation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every accepted relation specifies the assumptions and applicable subset.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
