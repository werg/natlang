---
description: Calculate absolute reminder times from exact appointment timestamps and approved relative offsets.
args: { dossier: PreparationDossier; change: AppointmentChange; requirements: PreparationRequirement[] }
returns: RelativeReminder[]
---
Use a deterministic time service with the supplied timezone, appointment timestamp and integer-minute offset. Return the computed absolute reminder timestamp and retain both versions and source IDs. Do not approximate offsets or infer daylight-saving transitions from local prose; block calculation if timezone data is incomplete.
