---
args: { candidate: RoleInterface, decisions: DecisionsIntegrationWave, evidence: EvidencePage[] }
returns: DraftIntegrationWave
---
1. Construct an evidence-backed IntegrationWave record from established atoms, using the fields declared for IntegrationWave in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Group compatible vocabulary and procedure changes into reversible waves. Preserve source identities, specify conditional transformations, and defer records whose meaning cannot be preserved under the target model without an explicit extension.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
