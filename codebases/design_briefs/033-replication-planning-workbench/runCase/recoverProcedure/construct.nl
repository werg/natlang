---
args: { candidate: ReplicationProject, decisions: DecisionsMethodStep, evidence: EvidencePage[] }
returns: DraftMethodStep
---
1. Construct an evidence-backed MethodStep record from established atoms, using the fields declared for MethodStep in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Reconstruct the reported procedure from methods, supplements, code descriptions, and notes. Separate explicitly specified parameters from inferred defaults and distinguish the authors’ actual method from a convenient modern replacement.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
