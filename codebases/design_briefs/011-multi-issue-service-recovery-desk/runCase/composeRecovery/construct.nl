---
args: { candidate: EntitlementFinding, decisions: DecisionsRemedyBundle, evidence: EvidencePage[] }
returns: DraftRemedyBundle
---
1. Construct an evidence-backed RemedyBundle record from established atoms, using the fields declared for RemedyBundle in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Construct alternative bundles that jointly solve the issues, accounting for dependencies and customer preferences. Detect double compensation for the same loss, but do not erase separate entitlements merely because they share one order.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
