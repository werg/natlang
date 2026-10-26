---
args: { candidate: MethodQuestion, decisions: DecisionsReplicationTask, evidence: EvidencePage[] }
returns: DraftReplicationTask
---
1. Construct an evidence-backed ReplicationTask record from established atoms, using the fields declared for ReplicationTask in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Translate the reviewed procedure into typed tasks using approved computational capabilities. Pin inputs and environments, define checkpoint and output contracts, and include resource limits and recovery behavior for failed jobs.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
