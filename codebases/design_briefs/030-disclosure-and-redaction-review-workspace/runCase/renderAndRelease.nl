---
args: { goal: GoalContract, policy: PolicySelection, candidates: DisclosureRisk[], evidence: EvidencePage[] }
returns: StageResultDisclosureManifest
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Render irreversible redactions through a trusted host service, inspect extracted text and metadata, and assemble a manifest. Bind approval and delivery to the exact recipient and artifact hashes, retaining a controlled review copy and disclosure receipt.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual DisclosureManifest fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No recoverable redacted text, hidden layers, or wrong-recipient delivery.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
