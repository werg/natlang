---
args: { candidate: Requirement, decisions: DecisionsAcceptanceScenario, evidence: EvidencePage[] }
returns: DraftAcceptanceScenario
---
1. Construct an evidence-backed AcceptanceScenario record from established atoms, using the fields declared for AcceptanceScenario in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Generate ordinary, boundary, and adverse scenarios, then replay the specification through the available simulator. Use counterexamples to reopen only the implicated agreements, preserving unrelated acceptance and recording revised stakeholder acknowledgment.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
