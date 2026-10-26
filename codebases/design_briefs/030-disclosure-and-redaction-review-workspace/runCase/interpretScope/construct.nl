---
args: { candidate: DisclosureCase, decisions: DecisionsDisclosureScope, evidence: EvidencePage[] }
returns: DraftDisclosureScope
---
1. Construct an evidence-backed DisclosureScope record from established atoms, using the fields declared for DisclosureScope in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read the request, authorization, and supplied disclosure policy together. Determine recipient, purpose, relevant time span, and permitted material; flag conflicting instructions rather than treating the broadest request as sufficient authority.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
