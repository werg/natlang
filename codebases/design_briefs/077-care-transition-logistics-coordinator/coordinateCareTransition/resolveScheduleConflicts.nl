---
description: Recompute downstream task windows after transport or appointment changes and expose access conflicts.
args: { dossier: TransitionDossier; change: TransitionChange; taskGraph: PrerequisiteState[] }
returns: ScheduleImpact[]
---
Call `compareTransitionWindows(dossier, change, taskGraph)` using the exact appointment, vehicle, destination access and task intervals supplied. Recompute only tasks downstream of a changed window. Preserve accessible-transport requirements. If a revised schedule cannot meet a clinical appointment or access window, return a conflict for coordinator escalation rather than altering the care plan.
