---
args: { goal: GoalContract, policy: PolicySelection, candidates: DisputeRecommendation[], evidence: EvidencePage[] }
returns: StageResultReviewDossier
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Assemble a balanced dossier with evidence, disputed questions, reasoning summary, and each party’s response opportunity. Redact unnecessary personal information while preserving contestable evidence and record the reviewer’s actual disposition separately from the recommendation.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ReviewDossier fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Both relevant positions represented; disclosure scope and response windows satisfied.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
