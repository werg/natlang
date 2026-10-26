---
args: { goal: GoalContract, policy: PolicySelection, candidates: Counteroffer[], evidence: EvidencePage[] }
returns: StageResultSourcingAgreement
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Interpret responses as acceptance, rejection, or qualified counteroffer at the term level. Rebuild package feasibility after each material change, identify unresolved conditions, and prepare a final agreement only when all linked commitments are mutually consistent.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual SourcingAgreement fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Qualified acceptance is not final acceptance; exact final package must match approvals.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
