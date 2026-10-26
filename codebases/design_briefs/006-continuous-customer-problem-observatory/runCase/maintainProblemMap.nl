---
args: { goal: GoalContract, policy: PolicySelection, candidates: IncidentGroup[], evidence: EvidencePage[] }
returns: StageResultProblemConcept
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Cluster by the failed outcome and mechanism rather than shared vocabulary. Propose new concepts when existing boundaries do not fit, preserve meaningful subtypes, and review counterexamples before moving historical episodes between concepts.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ProblemConcept fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Reclassification preserves raw episode IDs and records concept-version lineage.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
