---
args: { candidate: RedactionProposal, decisions: DecisionsDisclosureRisk, evidence: EvidencePage[] }
returns: DraftDisclosureRisk
---
1. Construct an evidence-backed DisclosureRisk record from established atoms, using the fields declared for DisclosureRisk in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Review the proposed bundle for information revealed by combining otherwise permissible fragments, filenames, metadata, and repeated identifiers. Distinguish plausible reconstruction from speculative risk and propose targeted changes for reviewer approval.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
