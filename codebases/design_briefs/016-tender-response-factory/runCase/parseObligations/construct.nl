---
args: { candidate: TenderResponse, decisions: DecisionsTenderRequirement, evidence: EvidencePage[] }
returns: DraftTenderRequirement
---
1. Construct an evidence-backed TenderRequirement record from established atoms, using the fields declared for TenderRequirement in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read the main tender, schedules, forms, and addenda together. Extract explicit requirements and conditional submission instructions, distinguish mandatory eligibility from scored preferences, and preserve exact formatting or attachment obligations separately from substantive answers.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
