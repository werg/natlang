---
args: { candidate: PurchaseIntent, decisions: DecisionsAgreementScope, evidence: EvidencePage[] }
returns: DraftAgreementScope
---
1. Construct an evidence-backed AgreementScope record from established atoms, using the fields declared for AgreementScope in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Resolve the applicable contract, schedules, amendments, and purchasing entity. Interpret coverage under dates, territory, and intended use; do not apply a parent agreement to an affiliate or service category without supporting terms.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
