---
args: { candidate: PolicyCaseFacts, decisions: DecisionsOutcomeDifference, evidence: EvidencePage[] }
returns: DraftOutcomeDifference
---
1. Construct an evidence-backed OutcomeDifference record from established atoms, using the fields declared for OutcomeDifference in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Execute both policy versions with isolated state and compare duties, permissions, timing, and escalation. Trace differences to specific changed conditions and separate genuine policy effects from nondeterministic or unsupported semantic judgments.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
