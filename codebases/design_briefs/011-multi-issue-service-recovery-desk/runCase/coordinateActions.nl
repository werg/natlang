---
args: { goal: GoalContract, policy: PolicySelection, candidates: RemedyBundle[], evidence: EvidencePage[] }
returns: StageResultRecoveryAction
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Sequence approved refunds, replacements, pickups, and communications. Reserve scarce replacements before promising dates, wait for required customer choices, and reconcile uncertain external outcomes rather than retrying irreversible operations blindly.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual RecoveryAction fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every effect has an operation identity and a matching approval or preauthorized policy grant.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
