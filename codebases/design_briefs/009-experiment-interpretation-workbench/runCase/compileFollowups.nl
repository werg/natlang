---
args: { goal: GoalContract, policy: PolicySelection, candidates: MechanismFinding[], evidence: EvidencePage[] }
returns: StageResultAnalysisOrder
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Convert unresolved issues into typed analysis plans with cohorts, time windows, allowed operations, and interpretation rules. Separate confirmatory follow-up from exploratory searching and require a fresh data or holdout plan where the proposed claim needs one.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AnalysisOrder fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Work orders compile against available schemas and expose every post-hoc choice.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
