---
description: Route unresolved preparation conflicts and missing exception authority to clinic staff.
args: { dossier: PreparationDossier; change: AppointmentChange; requirements: PreparationRequirement[]; conflicts: InstructionConflict[] }
returns: PreparationQuestion[]
---
For each conflict, identify the exact instruction versions, field or exception in dispute, and staff role authorized to clarify. Do not ask the patient to choose between clinical directions. Keep the affected checklist item blocked until an authoritative answer is recorded.
