---
args: { candidate: TenderRequirement, decisions: DecisionsCapabilityMatch, evidence: EvidencePage[] }
returns: DraftCapabilityMatch
---
1. Construct an evidence-backed CapabilityMatch record from established atoms, using the fields declared for CapabilityMatch in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare required outcomes with verified capabilities and their scope. Do not stretch an adjacent feature into compliance; identify exceptions, delivery dependencies, and evidence that would substantiate a currently unverified claim.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
