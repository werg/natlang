---
args: { candidate: ExceptionCase, decisions: DecisionsExceptionRecord, evidence: EvidencePage[] }
returns: DraftExceptionRecord
---
1. Construct an evidence-backed ExceptionRecord record from established atoms, using the fields declared for ExceptionRecord in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Extract the exact deviation, beneficiary, dates, and conditions of each granted exception. Keep the requirement that was waived separate from prerequisites that remained binding, and record whether the decision explicitly denied precedential value.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
