---
description: Execute generated semantic branches over independent event histories and inspect their observables.
args:
  inputs: ScenarioInput[]
  program: GeneratedProgram
  model: ProcedureModel
returns: ScenarioSuiteResult
---
1. For each original ScenarioInput initialize ScenarioFrontier with its initial atoms, actual event
   IDs ordered by explicit times, empty traces, the actual generated program and selected semantic model.
2. Invoke replayStep.iterateOn(frontier) with measure remainingEventIds.length until empty. Each step
   consumes one event and returns new actor/time-indexed state plus actual guard decision contributions.
3. Keep scenarios independent. A safety acknowledgment established in one history cannot be reused
   as state in another; they share governed procedure source, not executed event facts.
4. Call checkOutcome(input, completedFrontier, model). Consume its satisfied/violated observable
   lists, uncertainty and exact witnesses; a final generic pass label cannot replace the returned traces.
5. Return verdicts and unresolved. Revision targets arise from actual generated-program divergence,
   including missing state propagation or emitted guards that do not match their selected source plans.
6. Preserve adversarial scenarios even if the current generated candidate fails them. A failed source
   interpretation is a repair target, not permission to omit an inconvenient scenario from the suite.
