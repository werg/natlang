---
description: Prepare versioned appointment instructions, exact relative reminders and consent-bounded delivery state.
args: { dossier: PreparationDossier; change: AppointmentChange; coordinator: PreparationCoordinator }
returns: PreparationProposal
---
Use only the current authorized instruction and exact appointment time. Preserve conflicting patient-reported directions for clinic staff; do not choose which instruction is clinically correct.

Bind requirements = `recoverApprovedInstructions(dossier, change)` to bind each action, exception and timing rule to its source version. Bind contexts = `interpretPatientConstraints(dossier, change, requirements)` without changing clinical content.

Bind compilation = `compileRelativeChecklist(dossier, change, requirements, contexts)`. Recompute affected reminder times after appointment changes using exact supplied timezone and offsets. Preserve the prior checklist version and invalidate only reminders dependent on changed instructions or appointment time.

If translation changes a negation, exception or timing condition, block release until qualified fidelity review is supplied. Bind instructionConflicts = `identifyInstructionConflicts(dossier, change, requirements)`. Bind conflicts = `routeInstructionQuestions(dossier, change, requirements, instructionConflicts)` for clinical contradictions or missing exception authority.

Bind delivery = `trackPreparationDelivery(dossier, change, compilation.checklists)`. Verify current appointment-specific consent before sending to a representative. A delivered message is not acknowledged until a receipt says so. When delivery fails, use only the approved patient fallback channel.

Return `PreparationProposal` for `coordinator.id` with versioned reminders from `compilation.reminders`, instruction conflicts, translation findings, consent scope and a resumable delivery ledger. This service communicates approved instructions; it does not author clinical guidance.
