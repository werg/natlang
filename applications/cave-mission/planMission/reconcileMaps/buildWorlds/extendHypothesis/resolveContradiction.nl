---
description: Decide whether overlapping survey claims genuinely conflict under one proposed transform.
args:
  claims: PassageAssertion[]
  hypothesis: FrameHypothesis
  survey: SurveyEvidence
returns: AnchorAssessment
---
1. Compare the claims after applying hypothesis bindings, keeping their original frame and witnesses.
   Similar endpoint strings in different frames are not automatically the same physical corridor.
2. Distinguish existence, direction and carried-state permission. A present narrow tunnel and a rule
   blocking a full carrier describe compatible facts; they do not require deleting the tunnel.
3. Treat explicit one-way flow as a directional constraint. A physically adjacent return endpoint does
   not override a source statement that reverse travel is unavailable.
4. Read temporal scope before asserting inconsistency. The first fixture is explicitly a stable current
   world; do not invent a later water reversal to reconcile a forbidden return route.
5. Return compatible only when all claims can hold together under the interpreted scopes. Otherwise
   explain the exact conflicting premises with their witnesses so the rejected hypothesis is reviewable.
6. If a missing scope would decide compatibility, return compatible:null with that unresolved premise
   in reason; the caller must preserve it as unresolved rather than claiming the map was disproved.
