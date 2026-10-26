---
args: { candidate: IntegrationProject, decisions: DecisionsTermSense, evidence: EvidencePage[] }
returns: DraftTermSense
---
1. Construct an evidence-backed TermSense record from established atoms, using the fields declared for TermSense in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Extract terms from actual usage rather than glossary headings alone. Separate multiple senses within one organization and identify distinctions that affect who acts, which records exist, or what counts as completion.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
