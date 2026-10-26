---
args: { candidate: DisclosureRisk, decisions: DecisionsDisclosureManifest, evidence: EvidencePage[] }
returns: DraftDisclosureManifest
---
1. Construct an evidence-backed DisclosureManifest record from established atoms, using the fields declared for DisclosureManifest in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Render irreversible redactions through a trusted host service, inspect extracted text and metadata, and assemble a manifest. Bind approval and delivery to the exact recipient and artifact hashes, retaining a controlled review copy and disclosure receipt.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
