---
args: { candidate: ReplicationTask, decisions: DecisionsReplicationFinding, evidence: EvidencePage[] }
returns: DraftReplicationFinding
---
1. Construct an evidence-backed ReplicationFinding record from established atoms, using the fields declared for ReplicationFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare outputs with the original reported quantities and tolerances, distinguishing implementation failures from substantive differences. Investigate divergences through minimal controlled changes and publish uncertainty rather than claiming replication from superficial similarity.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
