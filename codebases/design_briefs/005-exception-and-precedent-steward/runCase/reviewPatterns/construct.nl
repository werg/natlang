---
args: { candidate: ExceptionDisposition, decisions: DecisionsPolicyReviewSignal, evidence: EvidencePage[] }
returns: DraftPolicyReviewSignal
---
1. Construct an evidence-backed PolicyReviewSignal record from established atoms, using the fields declared for PolicyReviewSignal in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Group substantively similar exceptions while discounting copied requests and a single repeated incident. Formulate a policy revision and boundary examples, then simulate historical outcomes before referring the proposal to the policy owner.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
