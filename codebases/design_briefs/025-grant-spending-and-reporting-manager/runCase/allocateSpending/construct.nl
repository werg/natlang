---
args: { candidate: GrantCondition, decisions: DecisionsGrantAllocation, evidence: EvidencePage[] }
returns: DraftGrantAllocation
---
1. Construct an evidence-backed GrantAllocation record from established atoms, using the fields declared for GrantAllocation in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Match expenditures to eligible activities using descriptions, service periods, and supporting records. Split shared costs only under documented allocation rules, detect duplicate funding attribution, and retain unsupported portions for review.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
