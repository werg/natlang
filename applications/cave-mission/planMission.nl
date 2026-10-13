---
description: Reconcile uncertain cave maps and produce an all-world mission policy.
kind: directory-reducer
args:
  request: MissionRequest
returns: AuditedMission
---
1. Call survey(folder, request) to read the immutable evidence snapshot and interpret frames, passages,
   equipment and sensor scopes. Preserve each returned category and its exact clause witnesses.
2. Call reconcileMaps(surveyResult). The returned WorldSet contains every surviving supported map,
   including alternative frame alignments that the current evidence does not distinguish.
3. Call designPolicy(request, surveyResult, worldsResult). It explores belief states and evaluates
   complete observation branches; it must never optimize for one guessed physical map.
4. Call auditMission(request, surveyResult, worldsResult, planningResult). Return its AuditedMission,
   including failed world replays and unresolved source propositions rather than hiding them.
5. All actions are proposed data. Do not operate a robot, alter the source files or contact an external
   sensor. Listed sensor outcomes are derived from the closed fictional world model.
6. Preserve source landmark, passage, sensor and clause identifiers. Generated world, state, action
   and policy-node IDs are arbitrary unique references; their spelling is not a semantic answer.
7. Distinguish an unknown corridor from a blocked corridor and a corridor known not to exist. An
   uncertain frame creates supported alternative worlds, not a default coordinate or a blocked edge.
8. Consume the children's actual typed results. A conflicting view belongs in audit; do not replace
   it with a new plan invented from raw source after the declared planning branch has returned.
