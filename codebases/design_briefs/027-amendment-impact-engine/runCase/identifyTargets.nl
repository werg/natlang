---
args: { goal: GoalContract, policy: PolicySelection, candidates: AmendmentCase[], evidence: EvidencePage[] }
returns: StageResultAmendmentTarget
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read amendment language against the referenced agreement edition. Resolve section references, defined terms, and indirect targets; distinguish correction of wording from a substantive change and retain ambiguous references for review rather than choosing the nearest heading.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AmendmentTarget fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Target identity and base edition established before transformation.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
