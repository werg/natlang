---
description: Compile a versioned checklist and exact relative reminder times for the current appointment.
args: { dossier: PreparationDossier; change: AppointmentChange; requirements: PreparationRequirement[]; contexts: PreparationContext[] }
returns: ChecklistCompilation
---
Call `computeRelativeReminders(dossier, change, requirements)` from exact appointment time, timezone and approved offsets. Keep each reminder linked to its instruction and appointment versions. Recompute only reminders whose source instruction or appointment changed. Preserve the old checklist for audit. If time zone or exact offset is missing, block the affected reminder rather than guessing.
