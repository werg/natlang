---
args: { candidate: ReviewPlan, decisions: DecisionsResolutionDelta, evidence: EvidencePage[] }
returns: DraftResolutionDelta
---
1. Construct an evidence-backed ResolutionDelta record from established atoms, using the fields declared for ResolutionDelta in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Reconcile owner responses against the pending question and current evidence version. Distinguish approval of wording from approval of the decision; append the adopted resolution and invalidate only consumers of the replaced version.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
