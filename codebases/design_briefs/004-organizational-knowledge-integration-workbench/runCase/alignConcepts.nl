---
args: { goal: GoalContract, policy: PolicySelection, candidates: TermSense[], evidence: EvidencePage[] }
returns: StageResultConceptAlignment
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare operational meanings and counterexamples for candidate terms. Establish equivalence only when obligations and consequences align; represent partial overlap explicitly and retain organization-specific variants where unification would lose behavior.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ConceptAlignment fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Bidirectional substitution tests pass before a same relation is accepted.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
