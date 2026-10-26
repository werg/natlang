---
description: Compare the supplied appointment, transport and access intervals without rounding schedule boundaries.
args: { dossier: TransitionDossier; change: TransitionChange; taskGraph: PrerequisiteState[] }
returns: ScheduleImpact[]
---
Use the source's actual time zone and interval boundaries. Identify overlaps between transport, appointments and destination access. Do not infer travel duration or silently round an appointment. Preserve original times and label missing duration or zone evidence as unresolved.
