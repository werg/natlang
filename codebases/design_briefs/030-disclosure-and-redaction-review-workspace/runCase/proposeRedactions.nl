---
args: { goal: GoalContract, policy: PolicySelection, candidates: DisclosureSelection[], evidence: EvidencePage[] }
returns: StageResultRedactionProposal
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Apply supplied rules to personal, confidential, privileged, or third-party material. Propose the smallest redaction that protects the restricted information while preserving the permitted claim; identify cases requiring qualified interpretation rather than making final legal judgments.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual RedactionProposal fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Each redaction has a rule and precise span; over-redaction is evaluated separately from leakage.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
