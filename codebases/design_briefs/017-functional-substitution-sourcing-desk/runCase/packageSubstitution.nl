---
args: { goal: GoalContract, policy: PolicySelection, candidates: QualificationTask[], evidence: EvidencePage[] }
returns: StageResultSubstitutionPacket
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare qualified alternatives on verified cost, lead time, and operational consequences. Present the recommended substitution with adaptations and residual uncertainty, obtain technical and commercial approval, and record which uses the approval does not cover.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual SubstitutionPacket fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Approval is scoped to intended use, lot or revision where relevant, and current evidence.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
