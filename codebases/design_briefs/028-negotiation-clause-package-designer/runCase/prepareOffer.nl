---
args: { goal: GoalContract, policy: PolicySelection, candidates: RedlineFinding[], evidence: EvidencePage[] }
returns: StageResultClauseOffer
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Assemble a coherent draft, issue list, and internal rationale with confidential fallback positions excluded from the external copy. Bind approval to the exact package and record which issues remain open rather than presenting a partial agreement as final.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ClauseOffer fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: External artifact contains no internal negotiation limits; current scoped approval required.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
