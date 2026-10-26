---
args: { goal: GoalContract, policy: PolicySelection, candidates: PurchaseIntent[], evidence: EvidencePage[] }
returns: StageResultAgreementScope
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Resolve the applicable contract, schedules, amendments, and purchasing entity. Interpret coverage under dates, territory, and intended use; do not apply a parent agreement to an affiliate or service category without supporting terms.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AgreementScope fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact temporal and entity scope plus supported semantic category match.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
