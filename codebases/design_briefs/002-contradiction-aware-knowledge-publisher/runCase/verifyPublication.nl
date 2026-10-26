---
args: { goal: GoalContract, policy: PolicySelection, candidates: PagePatch[], evidence: EvidencePage[] }
returns: StageResultPublicationCheck
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Re-extract claims from the proposed page set and repeat comparison around changed topics. Inspect whether simplification removed exceptions; build a review packet with residual disagreements and publish only after required owners approve the same patch set.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PublicationCheck fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Zero new contradiction in protected fixtures and no unresolved blocking conflict.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
