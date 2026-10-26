---
args: { goal: GoalContract, policy: PolicySelection, candidates: NegotiationPosition[], evidence: EvidencePage[] }
returns: StageResultClauseDependency
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Inspect definitions, remedies, caps, exceptions, and cross-references together. Identify where a concession changes another clause’s effect, including apparently local edits that broaden an exception or undermine a negotiated protection.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ClauseDependency fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Interaction claims have source references and a concrete example of changed behavior.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
