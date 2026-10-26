---
args: { goal: GoalContract, policy: PolicySelection, candidates: RedactionProposal[], evidence: EvidencePage[] }
returns: StageResultDisclosureRisk
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Review the proposed bundle for information revealed by combining otherwise permissible fragments, filenames, metadata, and repeated identifiers. Distinguish plausible reconstruction from speculative risk and propose targeted changes for reviewer approval.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual DisclosureRisk fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Bundle-level checks include metadata and cross-document references, not text alone.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
