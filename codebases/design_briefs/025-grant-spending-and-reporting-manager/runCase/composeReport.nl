---
args: { goal: GoalContract, policy: PolicySelection, candidates: ActivityEvidence[], evidence: EvidencePage[] }
returns: StageResultGrantNarrative
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Draft a coherent account aligned with actual spending, activities, and verified results. Explain deviations and unresolved conditions honestly, tailor the format to funder requirements, and cross-check narrative quantities against exact financial schedules.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual GrantNarrative fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No unsupported impact claim or mismatch between narrative and financial totals.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
