---
args: { candidate: ExpenseLine, decisions: DecisionsPurposeAllocation, evidence: EvidencePage[] }
returns: DraftPurposeAllocation
---
1. Construct an evidence-backed PurposeAllocation record from established atoms, using the fields declared for PurposeAllocation in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Identify mixed business and personal components using explicit facts and supplied allocation rules. Request a targeted explanation when purpose changes eligibility, and avoid inferring personal behavior from merchant stereotypes or unrelated account history.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
