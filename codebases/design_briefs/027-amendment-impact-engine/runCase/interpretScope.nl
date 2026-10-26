---
args: { goal: GoalContract, policy: PolicySelection, candidates: AmendmentTarget[], evidence: EvidencePage[] }
returns: StageResultAmendmentScope
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Determine effective dates, covered transactions, parties, and explicit exclusions. Separate signature date from effective date, preserve grandfathered cases, and identify whether the amendment applies retroactively only when the supplied text supports that interpretation.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AmendmentScope fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No application outside established temporal and entity scope.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
