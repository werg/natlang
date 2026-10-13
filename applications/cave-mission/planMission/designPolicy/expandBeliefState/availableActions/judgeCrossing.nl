---
description: Invoke a carried-state scope judgment for one directed passage in one world.
args:
  action: MissionAction
  robot: RobotState
  world: WorldModel
  gear: GearRule[]
returns: CrossingDecision
---
1. Select the exact directed WorldPassage matching action.from/to and its relevant gear clauses.
   If none exists, return allowed false with the explicit absence or missing-direction witnesses.
2. Capture that passage's condition, the relevant carried-state rules and robot's gear fields. In eval
   declare a local PassageJudgment with allowed:boolean, reason:string and witnesses:Witness[].
3. Create nl.with<PassageJudgment> over that scoped context and directly await it on the passage plus
   the current carried state. Ask whether this direction and clearance are permitted now, including
   sample bulk, correct fixed-rope attachment and one-way constraints.
4. Consume the actual returned allowed value. If false, return no nextRobot. If true, check the exact
   cost against remaining energy and subtract it; insufficient energy also returns allowed false.
5. A successful crossing changes location and energy only, unless a source clause states another
   physical transition. Traversing the fixed rope does not manufacture another unused spool.
6. Return worldId, allowed, nextRobot, reason and witnesses. A correct-looking map coordinate cannot
   override the local judgment about a full carrier or a forbidden water-gate direction.
7. This literal owns crossing applicability. The parent may not append a movement edge after an
   inapplicable or unresolved judgment merely because its destination is attractive.
