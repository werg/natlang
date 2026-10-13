---
description: Compare observed generated-state behavior with governed scenario obligations and forbidden inferences.
args:
  input: ScenarioInput
  completed: ScenarioFrontier
  model: ProcedureModel
returns: ScenarioVerdict
---
1. Read each required observable against the actual completed traces and source-selected rules.
   Confirm actor, event-time permission, custody/readiness separation and preserved uncertainty individually.
2. Check every model.requirements.forbiddenConsequence on the history, including unsupported waiver
   extension, retroactive permission and copying another actor's readiness at a handover.
3. An expected denial can be a valid scenario result. A true permission from an invented premise is
   invalid even when a later action eventually becomes allowed and the final state looks plausible.
4. Preserve missing decisive evidence as unresolved. A scenario's affirmative contradicting fact is
   a false premise; an absent fact cannot be upgraded to false merely to make the branch simpler.
5. Inventory satisfiedObservables and violatedObservables with exact event/source witnesses. Link a
   violation to the earliest actual guard/state effect that diverges rather than only its final summary.
6. Return scenarioId, valid, original traces, observable lists, unresolved and witnesses. The revision
   reducer consumes these concrete failures and does not rewrite the input's requested observables.

7. Inspect every actual GuardDecision.matchesSource and both state channels. A wrong emitted value
   or effect is a violated observable even if sourceRequiredValue is correct or a later effect restores
   the final state. Never replace the emitted trace with the source-required trace to obtain success.
