---
description: Apply finite basis alternatives and test them against independent anchors.
args:
  survey: SurveyEvidence
  landmarks: LandmarkPartition
returns: FrameHypothesis[]
---
1. Take each local FrameAlternative from survey.frames.alternatives. Do not include a second global
   identity frame as a separate uncertain world when it only describes the fixed coordinate convention.
2. For every positioned local landmark, compute global point = origin + local.x*uVector + local.y*vVector.
   These vector calculations are exact after the convention interpretation; they do not choose a convention.
3. Match computed points to explicitly fixed global coordinates and same-place anchors. Call
   assessAnchorPair(localMention, globalMention, transformedPoint, landmarks) for each proposed binding.
4. Consume compatible anchor assessments to build bindings. Reject a transform only for a supported
   contradiction; an unresolved blue-circle match supplies no contradiction and no extra equality constraint.
5. Preserve logical chamber identities for places with no independently named global counterpart.
   Their computed positions can vary across hypotheses while the local source label remains the same.
6. Retain every hypothesis satisfying all established anchors. Attach transform and binding witnesses,
   including the clause that leaves the export selector unknown.
7. Return the finite hypotheses. No nearest-neighbor heuristic, file order or visual preference may
   silently resolve the still-supported reflection.
