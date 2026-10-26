---
args: { candidate: PolicySimulation, decisions: DecisionsPolicyDelta, evidence: EvidencePage[] }
returns: DraftPolicyDelta
---
1. Construct an evidence-backed PolicyDelta record from established atoms, using the fields declared for PolicyDelta in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare baseline and proposed policies at the level of conditions, exceptions, authority, and consequences. Separate editorial changes from changed behavior and identify ambiguous provisions that prevent a determinate comparison.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
