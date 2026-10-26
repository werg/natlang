---
args: { goal: GoalContract, policy: PolicySelection, candidates: TenderRequirement[], evidence: EvidencePage[] }
returns: StageResultCapabilityMatch
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare required outcomes with verified capabilities and their scope. Do not stretch an adjacent feature into compliance; identify exceptions, delivery dependencies, and evidence that would substantiate a currently unverified claim.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual CapabilityMatch fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Unsupported certifications and future capabilities cannot be represented as current compliance.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
