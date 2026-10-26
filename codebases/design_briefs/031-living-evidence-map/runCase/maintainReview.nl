---
args: { goal: GoalContract, policy: PolicySelection, candidates: EvidenceClaim[], evidence: EvidencePage[] }
returns: StageResultEvidenceUpdate
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Process new reports, corrections, and retractions as scoped changes. Re-evaluate linked studies and dependent claims, preserve prior review snapshots, and request expert review where the new evidence changes methodological interpretation rather than merely adding another citation.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual EvidenceUpdate fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Retractions and corrections invalidate all affected derivatives; unchanged claims retain provenance.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
