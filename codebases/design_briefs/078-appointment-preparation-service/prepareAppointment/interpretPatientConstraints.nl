---
description: Record patient-stated communication, access and preparation constraints without changing clinical content.
args: { dossier: PreparationDossier; change: AppointmentChange; requirements: PreparationRequirement[] }
returns: PreparationContext[]
---
Bind each constraint to its source and the appointment. Preserve preferred language, accessible format and safe contact channel. A patient-reported contradiction with an approved instruction becomes an `InstructionConflict` for clinic staff; never resolve it by interpreting the clinical content yourself.
