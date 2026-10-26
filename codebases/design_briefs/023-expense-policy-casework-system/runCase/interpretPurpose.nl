---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExpenseLine[], evidence: EvidencePage[] }
returns: StageResultPurposeAllocation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Identify mixed business and personal components using explicit facts and supplied allocation rules. Request a targeted explanation when purpose changes eligibility, and avoid inferring personal behavior from merchant stereotypes or unrelated account history.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PurposeAllocation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Shares sum exactly to one; allocation has evidence or remains pending.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
