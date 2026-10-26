---
args: { candidate: ApplicabilityFinding, decisions: DecisionsExceptionDisposition, evidence: EvidencePage[] }
returns: DraftExceptionDisposition
---
1. Construct an evidence-backed ExceptionDisposition record from established atoms, using the fields declared for ExceptionDisposition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Construct a scoped recommendation with the original rule, proposed deviation, continuing controls, and expiration. Explain consistency with comparable cases and explicitly surface any departure requiring a higher authority.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
