---
description: Build a dependency graph from logistics tasks and unlock work only when prerequisite receipts exist.
args: { dossier: TransitionDossier; change: TransitionChange; requirements: TransitionRequirement[]; participants: TransitionParticipant[] }
returns: PrerequisiteState[]
---
Call `bindCompletionReceipts(dossier, change, requirements, participants)` and attach each receipt only to the task, equipment or person it actually confirms. Keep booked-but-undelivered equipment blocked. A caregiver's acceptance does not satisfy equipment delivery, transport or recipient acknowledgment. Preserve completed tasks separately from tasks merely scheduled.
