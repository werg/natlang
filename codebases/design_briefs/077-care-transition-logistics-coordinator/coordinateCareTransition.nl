---
description: Coordinate a care transition by unlocking logistical steps only after prerequisites and handoff receipts.
args: { dossier: TransitionDossier; change: TransitionChange; coordinator: TransitionCoordinator }
returns: TransitionProposal
---
Treat the approved care plan as authoritative for clinical requirements. This operation coordinates logistics; it does not change care instructions.

Bind requirements = `extractTransitionRequirements(dossier, change)` and bind each logistical outcome, prerequisite and responsible role. Bind participants = `mapAcceptedParticipants(dossier, change, requirements)`; a named caregiver is not available until they explicitly accept the role and its time window.

Bind taskGraph = `buildPrerequisiteSequence(dossier, change, requirements, participants)`. Keep each task blocked until required equipment delivery, responsible-person acceptance, transport, access and appointment prerequisites have source evidence. Equipment booked but lacking a delivery receipt does not unlock dependent work.

Bind schedule = `resolveScheduleConflicts(dossier, change, taskGraph)` after any transport or appointment change. Recompute affected downstream windows and accessible routes. Do not alter the approved clinical plan to make a schedule fit.

Bind handoff = `verifyTransitionHandoff(dossier, change, taskGraph, schedule)`. If a substitution affects a clinical requirement, keep it blocked until current clinician review. Require each dependency's completion evidence and receiving-organization acknowledgment before handoff readiness.

Return `TransitionProposal` for `coordinator.id` with the last completed task, blocked prerequisites, appointment impacts, escalations and resumable dependency graph. Planning does not prove delivery, caregiver acceptance or handoff completion.
