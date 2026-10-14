---
description: Judge one sensor's actual answer categories, observation scope and physical effects.
args:
  unit: SurveyReadingUnit
  primaryClauses: SourceClause[]
  supportingClauses: SourceClause[]
returns: SensorItemMeaning
---
1. Preserve this sensor unit and original source evidence. Capture its supporting scope; in eval
   create a scoped NL literal returning SensorItemMeaning and directly await it on the actual primary
   clauses and focus. No chosen physical world or expected mission observation enters the judgment.
2. Interpret sensor identity, available location, source cost and complete stated answer categories.
   Keep each answer's stated meaning distinct from its surface label. Missing category or scope
   information remains unresolved rather than expanding the sensor's capability.
3. Distinguish surface pigment, deep chamber structure and source-defined other properties. A repeated
   visible mark does not establish a physical landing identity or a route-relevant frame convention.
   Fixed-map direction is not replaced by the robot's current viewing direction.
4. Preserve whether observation changes location, gear or map, whether repeated answers are consistent,
   and whether repeated queries incur cost. Those facts belong in scope; they are not guessed effects.
5. Return a nullable sensor meaning and nonempty clause-keyed exact evidence. Every established cost,
   answer list and capability must be source-backed. Missing decisive evidence stays explicit unresolved.
6. Consume and return that actual result. Parent binding/checking supplies original Witness identity;
   later predictObservation invokes a separate NL judgment in each supported world to derive answers.
