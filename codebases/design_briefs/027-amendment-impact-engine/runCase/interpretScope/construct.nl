---
args: { candidate: AmendmentTarget, decisions: DecisionsAmendmentScope, evidence: EvidencePage[] }
returns: DraftAmendmentScope
---
1. Construct an evidence-backed AmendmentScope record from established atoms, using the fields declared for AmendmentScope in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Determine effective dates, covered transactions, parties, and explicit exclusions. Separate signature date from effective date, preserve grandfathered cases, and identify whether the amendment applies retroactively only when the supplied text supports that interpretation.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
