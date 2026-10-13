---
description: Judge one sensor answer against the feature it can actually observe in one physical world.
args:
  sensor: SensorRule
  world: WorldModel
  robot: RobotState
returns: SensorWorldAnswer
---
1. Select the world's bindings and facts governing the feature inside sensor.scope. Keep the fixed
   global orientation distinct from robot-facing direction, and deep structure distinct from surface marks.
2. Capture those premises and the source answer descriptions. In eval declare a local SensorJudgment
   with answer:string|null, reason:string and witnesses:Witness[]. Create nl.with<SensorJudgment>
   and directly await it with this one sensor/world feature question as changing input.
3. Ask the literal which declared answer follows in this world under the actual sensor scope. A
   nondiscriminating surface detection must not acquire an inference about the unseen deep chamber.
4. Consume the returned answer and witnesses. An answer outside the listed categories is unresolved;
   return null with the missing interpretation rather than inventing a more informative output category.
5. Return worldId with that answer. Do not choose a favorable world or modify robot gear/location;
   predicting an observation is a partition computation over already supported physical models.
6. The parent groups these actual decisions into cells. Different world IDs alone are not a sensor
   distinction, while the same answer in two physically different worlds must keep those worlds together.
