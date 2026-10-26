---
args: { goal: GoalContract, policy: PolicySelection, candidates: CloseTask[], evidence: EvidencePage[] }
returns: StageResultWorkpaperConclusion
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Assemble the reconciliation, explanations, and supporting documents into a reviewer-ready workpaper. Distinguish verified facts from management explanations, quantify residual differences exactly, and avoid conclusions broader than the evidence supports.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual WorkpaperConclusion fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Totals reconcile; residual material issues remain visible and block unqualified completion.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
