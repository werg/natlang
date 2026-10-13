---
description: Join landmarks and frame alternatives into all supported physical worlds.
args:
  survey: SurveyEvidence
returns: WorldSet
---
1. Call bindLandmarks(survey) to distinguish actual same-place anchors from reused visual marks.
2. Call alignFrames(survey, landmarkResult). It applies each supported transform and checks shared
   anchors while preserving alternative alignments not ruled out by the supplied evidence.
3. Call buildWorlds(survey, hypothesisResults). Each surviving frame hypothesis becomes a physical
   passage model or a witnessed rejection; world construction cannot simply choose the first transform.
4. Return the constructed worlds and rejections together with the landmark partition. Carry source
   interpretation unknowns that would change passage legality or observation predictions into unresolved.
5. A supported world is not a preferred route. Keep a difficult but consistent map for the planner;
   dropping it would weaken the universal mission requirement.
6. Merge models only when their physical bindings, directed passages, gear consequences and sensor
   outcomes agree. Equal-looking diagrams alone do not establish observational or task equivalence.
7. Keep generated world IDs stable within this result. Later belief states refer to these exact IDs,
   and every removal from a belief must be explained by an actual observation outcome.
