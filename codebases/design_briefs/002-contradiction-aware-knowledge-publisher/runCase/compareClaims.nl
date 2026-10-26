---
args: { goal: GoalContract, policy: PolicySelection, candidates: ScopedClaim[], evidence: EvidencePage[] }
returns: StageResultClaimConflict
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Generate candidates through shared entities and facets, then compare meanings under aligned scope and time. Classify coexistence as legitimate when claims describe different editions or populations; record the minimum incompatible propositions when a real conflict exists.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ClaimConflict fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Pair labels checked against scoped truth fixtures; duplicate pages add no authority.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
