---
args: { candidate: OwnershipTransfer, decisions: DecisionsContinuityReview, evidence: EvidencePage[] }
returns: DraftContinuityReview
---
1. Construct an evidence-backed ContinuityReview record from established atoms, using the fields declared for ContinuityReview in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Revisit the portfolio after departure, follow missing acknowledgments, and compare completed artifacts to promises. Close the handover only when ownership and immediate next steps are established, while operational work may legitimately remain open.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
