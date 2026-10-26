---
args: { candidate: DisclosureSelection, decisions: DecisionsRedactionProposal, evidence: EvidencePage[] }
returns: DraftRedactionProposal
---
1. Construct an evidence-backed RedactionProposal record from established atoms, using the fields declared for RedactionProposal in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Apply supplied rules to personal, confidential, privileged, or third-party material. Propose the smallest redaction that protects the restricted information while preserving the permitted claim; identify cases requiring qualified interpretation rather than making final legal judgments.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
