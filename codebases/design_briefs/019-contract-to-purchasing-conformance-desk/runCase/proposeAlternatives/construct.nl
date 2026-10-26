---
args: { candidate: ConformanceFinding, decisions: DecisionsPurchasingOption, evidence: EvidencePage[] }
returns: DraftPurchasingOption
---
1. Construct an evidence-backed PurchasingOption record from established atoms, using the fields declared for PurchasingOption in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Construct alternatives that satisfy the underlying need within the applicable agreement. Explain changed capabilities and total costs, and offer an exception route where compliant options would not meet essential requirements rather than silently weakening the request.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
