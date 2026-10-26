---
args: { goal: GoalContract, policy: PolicySelection, candidates: GrantReport[], evidence: EvidencePage[] }
returns: StageResultGrantCondition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Parse the agreement, budget schedules, and amendments into spending, milestone, and reporting conditions. Distinguish required outcomes from aspirational impact language, and preserve restrictions that apply only to specific funding tranches or time periods.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual GrantCondition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every condition has effective scope and an attributable grant source.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
