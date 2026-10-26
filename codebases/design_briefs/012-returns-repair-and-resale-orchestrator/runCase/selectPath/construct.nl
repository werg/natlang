---
args: { candidate: ConditionAssessment, decisions: DecisionsReturnDisposition, evidence: EvidencePage[] }
returns: DraftReturnDisposition
---
1. Construct an evidence-backed ReturnDisposition record from established atoms, using the fields declared for ReturnDisposition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Apply purchase terms and approved product-handling rules to condition evidence and customer preference. Identify cases needing specialist inspection, compare repair and replacement options, and avoid promising a remedy before required checks are satisfied.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
