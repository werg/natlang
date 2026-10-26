---
args: { goal: GoalContract, policy: PolicySelection, candidates: TenderResponse[], evidence: EvidencePage[] }
returns: StageResultTenderRequirement
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read the main tender, schedules, forms, and addenda together. Extract explicit requirements and conditional submission instructions, distinguish mandatory eligibility from scored preferences, and preserve exact formatting or attachment obligations separately from substantive answers.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual TenderRequirement fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every source requirement maps to a response slot or an explicit nonresponse decision.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
