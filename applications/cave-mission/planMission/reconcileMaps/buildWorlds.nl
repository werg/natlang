---
description: Construct and retain each consistent physical map through a shrinking hypothesis frontier.
args:
  survey: SurveyEvidence
  hypotheses: FrameHypothesis[]
returns: WorldConstruction
---
1. Initialize WorldFrontier with pending=hypotheses, empty worlds/rejected/unresolved and the interpreted survey.
2. Invoke extendHypothesis.iterateOn(initialState) with measure state.pending.length until pending
   is empty. Each step consumes one transform and owns its complete physical-model or rejection delta.
3. Preserve all returned worlds. Equal source counts or identical sensor names do not justify merging
   worlds whose correct anchor side or route differs.
4. Return WorldConstruction containing worlds, rejected and accumulated unresolved information.
   reconcileMaps joins this construction with its separately judged landmark partition.
5. If no model survives, report the contradictory premise set as unresolved rather than returning a
   vacuously successful all-world mission. At least one supported physical world is required for planning.
