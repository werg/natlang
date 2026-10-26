---
args: { candidate: PolicyDelta, decisions: DecisionsPolicyCaseFacts, evidence: EvidencePage[] }
returns: DraftPolicyCaseFacts
---
1. Construct an evidence-backed PolicyCaseFacts record from established atoms, using the fields declared for PolicyCaseFacts in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Normalize historical or synthetic cases into evidence-backed facts without adding missing information. Preserve uncertainty, policy-relevant context, and data-access restrictions; do not infer sensitive attributes merely to complete a simulation.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
