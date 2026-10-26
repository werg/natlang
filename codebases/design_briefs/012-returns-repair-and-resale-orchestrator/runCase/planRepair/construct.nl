---
args: { candidate: ReturnDisposition, decisions: DecisionsRepairPlan, evidence: EvidencePage[] }
returns: DraftRepairPlan
---
1. Construct an evidence-backed RepairPlan record from established atoms, using the fields declared for RepairPlan in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Translate the selected path into authorized inspection or repair tasks using supplied procedures. Check part compatibility and workshop capacity, preserve stop conditions, and create alternative plans when repair would exceed delegated cost or capability.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
