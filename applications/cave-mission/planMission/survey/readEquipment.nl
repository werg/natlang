---
description: Consume scoped per-item NL resource transitions and sensor capabilities.
args:
  clauses: SourceClause[]
returns: EquipmentSurvey
---
1. Capture immutable original clauses. Create a scoped NL inventory literal returning SurveyUnitInventory
   and directly await it. It identifies individual gear condition/transition and sensor reading units,
   primary/supporting original clause IDs and focus questions, with originalLocalName:null for these nonlandmark units; it supplies no prepared semantic rows.
2. Include initial resources, sample restrictions, rope rigging/reuse, permitted recovery and distinct
   sensor capabilities. Split several semantic items in one clause when their ownership differs; join
   one sensor's complementary answer/scope/cost clauses by actual source identity, not file adjacency.
3. Require nonempty primaryClauseIds, only gear/sensor kinds, existing original IDs and unique unit IDs.
   An invalid inventory reports its unit and expected source/kind for repair. Visit every actual unit.
   For gear call interpretGear(unit, primaryClauses, supportingClauses); for a sensor call
   interpretSensor(unit, primaryClauses, supportingClauses). Await and retain each actual child result.
4. Construct GearRule or SensorRule only from its returned nullable meaning. Conditions, consequences,
   scope, costs and answer descriptions come from that literal, not a parent-created case-specific table.
   A child's unresolved premise stays unresolved; missing meaning is never replaced with a predicted row.
5. Every established meaning requires nonempty clause-keyed exact evidence. Validate quotations within
   actual supplied original clause witnesses and copy original sourceId/path in the parent. A failed
   evidence check fails this contribution for repair before later planning can consume it.
6. Preserve conflicting source meanings; merge exact duplicate rows only. Join returned sensor categories
   only when original identity and scope agree. Observation effects and resource transitions retain
   different meanings even when both happen to cost one unit in this particular source world.
7. Return actual gear, sensors and inventory/child unresolved findings. Planning consumes these typed
   resource and observation rules; this stage neither selects an action nor predicts a sensor answer
   for a chosen physical world. No resource is changed while reading the snapshot.
