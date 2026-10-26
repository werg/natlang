---
args: { candidate: CompromiseOption, decisions: DecisionsRequirement, evidence: EvidencePage[] }
returns: DraftRequirement
---
1. Construct an evidence-backed Requirement record from established atoms, using the fields declared for Requirement in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Write atomic, testable requirements from accepted positions and compromises. Define terms, negative requirements, and failure behavior; keep unresolved choices outside the committed baseline and link every clause to its negotiated source.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
