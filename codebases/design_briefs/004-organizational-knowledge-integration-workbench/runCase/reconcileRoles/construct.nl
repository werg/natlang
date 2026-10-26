---
args: { candidate: ConceptAlignment, decisions: DecisionsRoleInterface, evidence: EvidencePage[] }
returns: DraftRoleInterface
---
1. Construct an evidence-backed RoleInterface record from established atoms, using the fields declared for RoleInterface in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare actual decision rights, escalation paths, and handoff expectations. Identify duplicated ownership and gaps between roles; propose responsibility agreements without assuming that similar job titles imply interchangeable authority.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
