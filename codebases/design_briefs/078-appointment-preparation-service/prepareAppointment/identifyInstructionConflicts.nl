---
description: Extract source-bound contradictions among approved instructions for staff resolution.
args: { dossier: PreparationDossier; change: AppointmentChange; requirements: PreparationRequirement[] }
returns: InstructionConflict[]
---
Compare only instructions whose source versions are current and applicable to this appointment. Record each conflicting statement with its source identifiers and the exact exception or timing field in dispute. A patient-reported account remains attributed to that account; it does not override an approved instruction. Do not resolve the clinical question here.
