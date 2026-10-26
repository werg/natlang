---
args: { goal: GoalContract, policy: PolicySelection, candidates: GrantAllocation[], evidence: EvidencePage[] }
returns: StageResultActivityEvidence
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare project records with milestone definitions and acceptance evidence. Distinguish activity completion from demonstrated impact, preserve partial delivery, and request missing evidence rather than embellishing a narrative to fit the grant’s aspirations.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ActivityEvidence fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Claimed outputs and outcomes are separately supported.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
