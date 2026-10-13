---
description: Build the actual observation partition by consuming scoped sensor meanings for every world.
args:
  action: MissionAction
  belief: BeliefState
  worlds: WorldSet
  survey: SurveyEvidence
returns: ObservationPartition
---
1. Find the SensorRule named by action.sensorId and require its availability at belief.robot.location.
   If the source does not authorize it here, return incomplete with the missing premise in reason.
2. For each world in belief.worldIds, call interpretSensorScope(sensor, world, belief.robot). Consume
   its actual SensorWorldAnswer rather than grouping models by convenient names or coordinates.
3. Group world IDs by equal returned answer categories, combining exact witnesses. A null answer
   leaves the partition incomplete; it cannot be silently assigned to a guessed outcome cell.
4. Check each input world appears in exactly one nonempty cell and no outside world appears. Preserve
   all cells, including those whose downstream mission will fail under the current gear commitment.
5. Set discriminating when the complete partition has more than one cell. A one-cell surface answer
   may be accurate while giving no information about the unknown chamber alignment.
6. Return sensorId, cells, complete, discriminating, reason and witnesses. The parent consumes these
   cells as the only observation successors and charges the sensor's stated cost in every outcome.
