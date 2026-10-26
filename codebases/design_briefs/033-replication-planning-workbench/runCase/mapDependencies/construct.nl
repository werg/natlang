---
args: { candidate: MethodStep, decisions: DecisionsReplicationDependency, evidence: EvidencePage[] }
returns: DraftReplicationDependency
---
1. Construct an evidence-backed ReplicationDependency record from established atoms, using the fields declared for ReplicationDependency in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Identify data, software, configuration, and computational dependencies for each step. Detect unavailable versions and incompatible assumptions, then formulate approved alternatives without representing them as exact replication of the original environment.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
