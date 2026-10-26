---
args: { goal: GoalContract, policy: PolicySelection, candidates: ClauseDependency[], evidence: EvidencePage[] }
returns: StageResultClausePackage
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Draft materially distinct packages that exchange permitted concessions while preserving required protections. Explain package-level tradeoffs and include fallback options; avoid optimizing individual clauses independently when their effects depend on the surrounding agreement.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ClausePackage fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: All hard playbook limits pass and no protection is claimed when another clause negates it.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
