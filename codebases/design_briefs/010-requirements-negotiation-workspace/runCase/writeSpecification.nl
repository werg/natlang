---
args: { goal: GoalContract, policy: PolicySelection, candidates: CompromiseOption[], evidence: EvidencePage[] }
returns: StageResultRequirement
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Write atomic, testable requirements from accepted positions and compromises. Define terms, negative requirements, and failure behavior; keep unresolved choices outside the committed baseline and link every clause to its negotiated source.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual Requirement fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No orphan requirement, hidden assumption, or mutually inconsistent accepted clause.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
