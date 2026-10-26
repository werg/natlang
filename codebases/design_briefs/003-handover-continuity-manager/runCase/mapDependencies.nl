---
args: { goal: GoalContract, policy: PolicySelection, candidates: WorkStatus[], evidence: EvidencePage[] }
returns: StageResultHandoverDependency
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Identify knowledge, access, approvals, and external relationships needed to finish each item. Separate credentials from instructions, detect circular handoffs, and formulate minimal access requests without copying secrets into the handover dossier.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual HandoverDependency fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No plaintext credentials; every blocked dependency has an owner and a recovery task.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
