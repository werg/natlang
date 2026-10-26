---
args: { candidate: AgreementScope, decisions: DecisionsConformanceFinding, evidence: EvidencePage[] }
returns: DraftConformanceFinding
---
1. Construct an evidence-backed ConformanceFinding record from established atoms, using the fields declared for ConformanceFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare each requested line with contract inclusions, exclusions, and conditions. Separate commercial price compliance from functional scope and preserve conditions such as minimum quantities, required bundles, or approved use limitations.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
