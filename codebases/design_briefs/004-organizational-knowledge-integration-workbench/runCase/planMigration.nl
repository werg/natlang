---
args: { goal: GoalContract, policy: PolicySelection, candidates: RoleInterface[], evidence: EvidencePage[] }
returns: StageResultIntegrationWave
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Group compatible vocabulary and procedure changes into reversible waves. Preserve source identities, specify conditional transformations, and defer records whose meaning cannot be preserved under the target model without an explicit extension.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual IntegrationWave fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact referential-integrity checks and semantic round-trip tests on representative records.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
