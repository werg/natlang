---
args: { candidate: SupplierIncident, decisions: DecisionsIncidentCluster, evidence: EvidencePage[] }
returns: DraftIncidentCluster
---
1. Construct an evidence-backed IncidentCluster record from established atoms, using the fields declared for IncidentCluster in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare complaints, inspections, and shipment events by product, lot, timing, and failure mechanism. Group plausibly related events while retaining independent incidents and contradictory evidence; do not merge all problems from one supplier into a single cause.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
