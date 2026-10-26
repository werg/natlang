---
args: { candidate: CodeDefinition, decisions: DecisionsCodingDecision, evidence: EvidencePage[] }
returns: DraftCodingDecision
---
1. Construct an evidence-backed CodingDecision record from established atoms, using the fields declared for CodingDecision in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Code bounded passages with their necessary context, then compare independent assessments on a sampled overlap. Investigate disagreements as possible codebook defects; when a definition changes, mark all affected earlier decisions stale before recoding.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
