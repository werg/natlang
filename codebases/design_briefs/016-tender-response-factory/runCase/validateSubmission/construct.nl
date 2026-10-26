---
args: { candidate: ResponseSection, decisions: DecisionsSubmissionManifest, evidence: EvidencePage[] }
returns: DraftSubmissionManifest
---
1. Construct an evidence-backed SubmissionManifest record from established atoms, using the fields declared for SubmissionManifest in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Re-run coverage and cross-section consistency, validate file naming and required forms, and preview the exact bundle. Freeze approvals to its hash, submit through an authorized adapter, and record portal acknowledgment separately from local upload completion.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
