---
args: { candidate: BoundaryScenario, decisions: DecisionsPolicyOption, evidence: EvidencePage[] }
returns: DraftPolicyOption
---
1. Construct an evidence-backed PolicyOption record from established atoms, using the fields declared for PolicyOption in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Summarize who and what changes under each candidate, then propose clarifications or alternatives for the policy owner. Preserve minority and exceptional cases, report coverage limits, and avoid choosing normative priorities not supplied by the organization.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
