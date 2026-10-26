---
args: { goal: GoalContract, policy: PolicySelection, candidates: AmendmentScope[], evidence: EvidencePage[] }
returns: StageResultAmendedObligation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Construct the amended obligation by applying the specific operation to its target. Preserve untouched exceptions, prerequisites, definitions, and cross-references; compare the result against counterexamples that would expose accidental broadening or deletion.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AmendedObligation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Protected conditions remain equivalent; every removed condition is authorized by the amendment.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
