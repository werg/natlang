---
args: { goal: GoalContract, policy: PolicySelection, candidates: ClaimConflict[], evidence: EvidencePage[] }
returns: StageResultAuthorityDecision
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Consult the configured publication hierarchy and explicit supersession records. Prefer authoritative material only inside its remit; if equally authoritative sources disagree, formulate a bounded clarification rather than selecting the newest or most fluent text.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AuthorityDecision fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Unresolved decisive authority blocks publication of the disputed claim.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
