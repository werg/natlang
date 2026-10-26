---
args: { candidate: ServiceIssue, decisions: DecisionsEntitlementFinding, evidence: EvidencePage[] }
returns: DraftEntitlementFinding
---
1. Construct an evidence-backed EntitlementFinding record from established atoms, using the fields declared for EntitlementFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Match each issue to the applicable policy edition, purchased service, and documented exception. Separate mandatory entitlements from discretionary gestures; ask for missing decisive evidence without requiring information already present in the case.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
