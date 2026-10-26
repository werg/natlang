---
description: Recover the current authorized preparation requirements without rewriting clinical instructions.
args: { dossier: PreparationDossier; change: AppointmentChange }
returns: PreparationRequirement[]
---
Use the instruction identified by the appointment record and verify its current version and authority. Preserve every required action, timing rule and exception as written. If newer sources conflict or an exception's meaning is uncertain, retain the current accepted version and route the conflict to staff rather than selecting an instruction.
