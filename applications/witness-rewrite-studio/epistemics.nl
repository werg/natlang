---
description: Construct character-local information states and dialogue licenses.
args:
  intake: IntakeResult
  world: WorldIndex
returns: KnowledgeIndex
---
1. Call acquire(intake, world). Consume its acquisitions and states as the information model.
   Knowledge follows a witnessed read, observation or credible communication with a prior informed source.
2. Call checkDialogue(intake, world, acquired). This deeper branch owns licenses for the actual
   propositions spoken by each actor, using the state immediately before the relevant event.
3. Keep acquisitions in the returned history even when their message is misleading. An actor's
   credible belief can motivate action without establishing that its content is an actual-world fact.
4. Retain explicit unresolved states where event order or message reception is unstated. Do not
   choose the chronology that happens to make all dialogue licensed.
5. Return acquisitions and states from acquire, and dialogue from checkDialogue, plus localDecisions drawn from their actual localDecision fields. Each dialogue
   decision carries the information path it consumed rather than only a verdict label.
6. A removed letter delivery can invalidate reading access while leaving sight of its seal unchanged.
   Preserve these different propositions; visual recognition of a sealed letter does not reveal contents.
7. When invoked on a candidate snapshot, reconstruct acquisitions from edited events and controlling
   canon. Never import the old state's knowledge merely because character names match.
8. The caller may use this branch to create repair obligations. It may not overwrite an unsupported
   decision with a plausible plot explanation absent from these returned sources.
