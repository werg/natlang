---
args: { goal: GoalContract, policy: PolicySelection, candidates: MeasureRelation[], evidence: EvidencePage[] }
returns: StageResultHarmonizationRule
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Select documented transformations or category mappings and describe lost distinctions. Delegate arithmetic to exact services, refuse unsupported imputation, and preserve source variables so analysts can inspect or reverse the transformation where possible.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual HarmonizationRule fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Transformation is defined for every accepted input state, including missing and out-of-range values.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
