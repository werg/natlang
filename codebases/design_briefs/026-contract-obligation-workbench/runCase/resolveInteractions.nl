---
args: { goal: GoalContract, policy: PolicySelection, candidates: ContractObligation[], evidence: EvidencePage[] }
returns: StageResultObligationRelation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Interpret cross-references and interactions among provisions under the supplied authority hierarchy. Distinguish a narrow exception from replacement of an entire obligation and preserve unresolved conflicts for qualified review instead of guessing a priority rule.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ObligationRelation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every relation has scope and source support; conflict is not silently converted into supersession.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
