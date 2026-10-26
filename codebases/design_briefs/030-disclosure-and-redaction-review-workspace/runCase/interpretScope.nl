---
args: { goal: GoalContract, policy: PolicySelection, candidates: DisclosureCase[], evidence: EvidencePage[] }
returns: StageResultDisclosureScope
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read the request, authorization, and supplied disclosure policy together. Determine recipient, purpose, relevant time span, and permitted material; flag conflicting instructions rather than treating the broadest request as sufficient authority.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual DisclosureScope fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Scope and recipient authority are established before document access or export.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
