---
args: { candidate: IncidentCluster, decisions: DecisionsExposureFinding, evidence: EvidencePage[] }
returns: DraftExposureFinding
---
1. Construct an evidence-backed ExposureFinding record from established atoms, using the fields declared for ExposureFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Trace potentially affected lots and orders through authoritative supply records. Distinguish confirmed exposure from precautionary scope, identify downstream commitments, and ask for missing traceability evidence before narrowing a containment boundary.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
