---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExceptionRecord[], evidence: EvidencePage[] }
returns: StageResultPrecedentCandidate
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Search by the rationale and relevant circumstances rather than matching requested outcomes. Construct analogies and disanalogies, including expired and rejected cases when they explain the boundary of acceptable treatment.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PrecedentCandidate fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Retrieval includes a negative precedent slice; similarity alone cannot authorize a waiver.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
