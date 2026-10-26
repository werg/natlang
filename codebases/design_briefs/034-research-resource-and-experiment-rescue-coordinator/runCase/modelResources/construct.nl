---
args: { candidate: ExperimentRequirement, decisions: DecisionsResourceCapability, evidence: EvidencePage[] }
returns: DraftResourceCapability
---
1. Construct an evidence-backed ResourceCapability record from established atoms, using the fields declared for ResourceCapability in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Reconcile calendars, instrument status, maintenance records, and operator qualifications. Distinguish booked availability from actual readiness and identify resources that appear interchangeable but differ in capability or validated use.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
