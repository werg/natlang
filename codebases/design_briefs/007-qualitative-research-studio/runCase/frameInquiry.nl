---
args: { goal: GoalContract, policy: PolicySelection, candidates: ResearchProject[], evidence: EvidencePage[] }
returns: StageResultInquiryFrame
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Interpret the decision that research should inform and decompose it into answerable questions. Identify plausible rival explanations and what the available corpus cannot establish; avoid converting an exploratory question into a predetermined conclusion.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual InquiryFrame fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every subquestion has observable evidence and an explicit inferential limit.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
