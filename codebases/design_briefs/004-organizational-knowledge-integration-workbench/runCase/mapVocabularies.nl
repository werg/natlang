---
args: { goal: GoalContract, policy: PolicySelection, candidates: IntegrationProject[], evidence: EvidencePage[] }
returns: StageResultTermSense
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Extract terms from actual usage rather than glossary headings alone. Separate multiple senses within one organization and identify distinctions that affect who acts, which records exist, or what counts as completion.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual TermSense fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every sense has contextual examples; identical spelling does not establish equivalence.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
