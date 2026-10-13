---
description: Generate physically legal action candidates and their deterministic state changes across the belief.
args:
  belief: BeliefState
  worlds: WorldSet
  survey: SurveyEvidence
returns: StateExpansion
---
1. Select the WorldModels named by belief.worldIds. Enumerate source-listed moves, rig sides, sensors
   at this location and pickup at the actual sample chamber. Never invent an unlisted action kind.
2. For each move candidate, call judgeCrossing(action, belief.robot, world, survey.gear) in each selected
   world. Retain the action only when every decision allows it and their nextRobot states agree.
3. Keep a rejected action with the first concrete failing world, rule and witnesses. If a route exists
   only under one map, it is not a safe unconditional move from a belief still containing both maps.
4. For rigging, require the source location, unused spool and enough energy. A wrong-side anchor can
   be physically rigged without revealing its usefulness; retain both physical possibilities for value search.
5. For pickup, require the sample location in every current world, an empty carrier and enough energy.
   Its successor has sampleHeld true. Apply the sealed-sample rule to future actions, not a hidden discard.
6. For observations, require the source sensor location and sufficient energy. Return a placeholder
   action edge with no outcomes; predictObservation will own its actual answer partition.
7. Give each action a unique ID, exact cost and witnessed source rule. Deterministic edges receive
   one null-answer outcome and a newStates entry for their computed successor RobotState/world set.
8. Return StateExpansion containing these candidates, rejected actions and deterministic successors.
   Its temporary nextState IDs are reconciled by the parent into the graph's canonical exact state keys.
9. Physical permission is separate from global success: rigging the wrong anchor may be legal but
   cannot survive the later all-world winning-value recurrence.
