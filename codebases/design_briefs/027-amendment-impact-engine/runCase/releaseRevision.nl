---
args: { goal: GoalContract, policy: PolicySelection, candidates: AmendmentImpact[], evidence: EvidencePage[] }
returns: StageResultAmendmentRelease
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Present a before-and-after obligation diff and affected-case replay to qualified reviewers. Freeze the approved interpretation, rebase active cases at controlled checkpoints, and preserve old versions for disputes, audit, and rollback of uncommitted changes.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AmendmentRelease fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No in-flight commitment changes silently; active cases record their interpretation migration decision.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
