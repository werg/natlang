---
args: { candidate: ProblemEpisode, decisions: DecisionsIncidentGroup, evidence: EvidencePage[] }
returns: DraftIncidentGroup
---
1. Construct an evidence-backed IncidentGroup record from established atoms, using the fields declared for IncidentGroup in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare accounts for the same customer, event, and time window. Group follow-ups and copied messages into one incident while retaining repeat failures that actually occurred on different occasions; leave identity ambiguity unresolved.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
